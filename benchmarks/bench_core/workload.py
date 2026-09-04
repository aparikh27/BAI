"""The perception-to-execution workload both arms run.

Everything above the robot driver is held constant across the comparison: the
same `Coordinator`, the same `PipelineManager`, the same real
`WebotsExecutorAgent` with its real `_get_object` servo logic. Only the driver
underneath differs — `EmberRobotDriver` over EMBER in one arm, the same class
over `LegacyRuntime` in the other (both clients expose an identical
synchronous surface, so the driver itself is literally the same code). Any
measured difference is therefore attributable to the runtime below the
driver and to nothing else.

**Why the perception agents are stubs.** Whisper transcription and YOLO
inference cost tens to hundreds of milliseconds and are byte-for-byte
identical in both arms — EMBER has no concept of vision, as ADR-005 states.
Including them would add the same large constant to both sides, which does
not change the difference but does swamp it: a 40us improvement inside a
300ms measurement is invisible, and the run time per iteration would make a
5000-iteration distribution impractical. The stubs preserve the pipeline's
real structure (four `Coordinator.dispatch` hops, real `Message` construction,
real payload merging) and take a configurable simulated cost, defaulting to
zero. `--perception-cost-ms` puts a realistic constant back in when the
question is "what fraction of a whole task did this improve", rather than
"how much faster is the part that changed".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agents.agent.base_agent import Agent
from agents.agent.execution_agent.webot_execution import WebotsExecutorAgent
from agents.master_planner.coordinator import Coordinator
from agents.messaging import Message, MessageStatus, MessageType

# The object is placed dead-centre in a 640px frame so
# WebotsExecutorAgent._align_and_approach's alignment loop is satisfied on its
# first check. That is deliberate: the servo loop's turn/re-observe cycle
# includes a 0.6s observation timeout that would dominate every sample and,
# worse, would vary with detection timing rather than with the runtime under
# test. Alignment is exercised (the check runs, `stop()` is issued) without
# its timing being folded into the result.
CAMERA_WIDTH_PX = 640
_CENTRED_BOX = (300.0, 100.0, 340.0, 140.0)


class BenchWorldObject:
    """Minimal object satisfying what `WebotsExecutorAgent` reads off a world
    object: `.track_id`, `.class_name`, `.visible`, `.box`, `.last_seen_frame`."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        visible: bool = True,
        box: tuple[float, float, float, float] = _CENTRED_BOX,
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.visible = visible
        self.box = box
        self.last_seen_frame = 0


class BenchWorldModel:
    """Static `WorldModel`. Static because a moving world would make each
    iteration take a different number of motor commands, and a latency
    distribution over a varying amount of work measures neither."""

    def __init__(self, objects: list[BenchWorldObject] | None = None):
        self.objects = objects if objects is not None else [
            BenchWorldObject(track_id=1, class_name="red ball"),
            BenchWorldObject(track_id=2, class_name="keys", box=(100.0, 50.0, 150.0, 80.0)),
        ]

    def get_visible_objects(self) -> list[BenchWorldObject]:
        return [obj for obj in self.objects if obj.visible]

    def get_object_by_track_id(self, track_id: int) -> BenchWorldObject | None:
        for obj in self.objects:
            if obj.track_id == track_id:
                return obj
        return None


def _burn(seconds: float) -> None:
    """Simulated perception cost.

    `time.sleep`, not a busy loop: a busy loop would hold the GIL and
    materially change how the asyncio client thread underneath gets scheduled,
    which would make the stub's cost interact with the thing being measured
    instead of merely adding to it.
    """
    if seconds > 0:
        time.sleep(seconds)


class StubAudioAgent(Agent):
    """Stands in for `WhisperAudioAgent`, returning a fixed transcription."""

    def __init__(self, cost_s: float = 0.0):
        super().__init__(name="Audio")
        self.cost_s = cost_s

    def handle_message(self, msg: Message) -> Message:
        _burn(self.cost_s)
        return self.create_response(
            request=msg,
            payload={"transcription": "get the red ball", "language": "en"},
        )


class StubVisionAgent(Agent):
    """Stands in for `YOLOVisionAgent`, returning one centred detection."""

    def __init__(self, cost_s: float = 0.0):
        super().__init__(name="Vision")
        self.cost_s = cost_s

    def handle_message(self, msg: Message) -> Message:
        _burn(self.cost_s)
        return self.create_response(
            request=msg,
            payload={
                "detections": [
                    {
                        "class_name": "red ball",
                        "confidence": 0.95,
                        "box": list(_CENTRED_BOX),
                        "track_id": 1,
                    }
                ]
            },
        )


class StubPlannerAgent(Agent):
    """Stands in for `QwenPlannerAgent`, emitting the plan the executor runs.

    A single `get_object` step, which `WebotsExecutorAgent._get_object`
    expands into eight motor commands (stop, move_forward, lower_arm,
    grab_item, raise_arm, turn, move_forward, turn) — a realistic
    manipulation task rather than a single-command microbenchmark wearing a
    pipeline as a costume.
    """

    MOTOR_COMMANDS_PER_TASK = 8

    def __init__(self, cost_s: float = 0.0, action: str = "get_object", target: str = "red ball"):
        super().__init__(name="Planner")
        self.cost_s = cost_s
        self.action = action
        self.target = target
        # Per-instance, not a class attribute: a mutable default at class
        # scope would accumulate alerts across every Coordinator built in a
        # single process, silently mixing one benchmark arm's relay traffic
        # into the next one's assertions.
        self.telemetry_alerts: list[dict[str, Any]] = []

    def handle_message(self, msg: Message) -> Message:
        if msg.action == "telemetry_alert":
            # The relay path (EmberTelemetryRelay -> Coordinator.dispatch)
            # dispatches here; acknowledging it keeps the relay's own dispatch
            # cost in the measurement instead of it short-circuiting on an
            # unregistered-action error.
            self.telemetry_alerts.append(msg.payload)
            return self.create_response(request=msg, payload={"acknowledged": True})

        _burn(self.cost_s)
        return self.create_response(
            request=msg,
            payload={"plan": [{"action": self.action, "target": self.target}]},
        )


@dataclass
class WorkloadConfig:
    perception_cost_s: float = 0.0
    camera_width_px: int = CAMERA_WIDTH_PX
    plan_action: str = "get_object"
    plan_target: str = "red ball"


def build_coordinator(robot_driver, config: WorkloadConfig | None = None) -> Coordinator:
    """Wires the four-stage pipeline over `robot_driver`.

    The Memory agent from the production wiring is left out: it writes to
    SQLite, and disk I/O variance would land in the tail of every sample
    while being entirely unaffected by which runtime executes motor commands.
    """
    cfg = config or WorkloadConfig()
    coordinator = Coordinator()
    coordinator.add_agent(StubAudioAgent(cfg.perception_cost_s))
    coordinator.add_agent(StubVisionAgent(cfg.perception_cost_s))
    coordinator.add_agent(StubPlannerAgent(cfg.perception_cost_s, cfg.plan_action, cfg.plan_target))
    coordinator.add_agent(
        WebotsExecutorAgent(robot_driver=robot_driver, world=BenchWorldModel())
    )
    return coordinator


def perception_payload(frame_index: int = 0) -> dict[str, Any]:
    """The synthetic stimulus: what a camera frame plus a speech command looks
    like entering the pipeline. Carries a real (if small) payload rather than
    an empty dict so that `Message` construction and the pipeline's
    `current_payload.update()` merge do the work they normally would."""
    return {
        "audio_path": "bench://speech/get_the_red_ball.wav",
        "image_path": "bench://camera/frame.png",
        "frame_index": frame_index,
        "captured_ns": time.perf_counter_ns(),
    }


def run_perception_to_execution(coordinator: Coordinator, frame_index: int = 0) -> Message:
    """One full Perception -> Execution pass.

    Returns the terminal `Message`, which is SUCCESS only once every motor
    command in the plan has been issued *and* acknowledged by the runtime —
    so the measured interval genuinely ends at "verified motor command frame"
    rather than at "command handed to a socket".
    """
    return coordinator.run_premade_pipeline("full_pipeline", perception_payload(frame_index))


def assert_pipeline_succeeded(result: Message) -> None:
    """Fails loudly on a pipeline that errored.

    Worth its own call: a failing pipeline still returns quickly, so a broken
    workload would otherwise show up as a spectacular latency improvement
    rather than as an error.
    """
    if result.status != MessageStatus.SUCCESS:
        raise RuntimeError(f"benchmark workload pipeline failed: {result.error!r}")
    if result.message_type != MessageType.RESPONSE:
        raise RuntimeError(f"unexpected terminal message type: {result.message_type!r}")


def motor_commands_per_iteration() -> int:
    return StubPlannerAgent.MOTOR_COMMANDS_PER_TASK
