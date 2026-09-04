"""Test rigs: one interface over the legacy and hybrid runtimes.

Each rig owns the full stack for its arm — runtime, bridge client, robot
driver, Coordinator, diagnostics — and exposes an identical surface, so every
benchmark in `compare_runtimes.py` is written once and executed twice. That
matters beyond tidiness: a benchmark written separately per arm is a
benchmark that can accidentally measure two different things, and the whole
report would be quietly wrong in a way no assertion would catch.

Both rigs are context managers. Startup and shutdown are genuinely tricky on
the hybrid side (a subprocess, a socket, a background event loop, a telemetry
thread) and leaking any of them across benchmark cases would contaminate the
next case's numbers, so the lifecycle is not left to the caller.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.agent.execution_agent.ember_bridge_client import EmberBridgeClient
from agents.agent.execution_agent.ember_execution_agent import EmberRobotDriver
from agents.agent.execution_agent.ember_telemetry_relay import EmberTelemetryRelay
from agents.messaging.ember_wire import AckResult, CommandOp, TelemetryFrame, TelemetrySubsystem
from agents.master_planner.coordinator import Coordinator
from benchmarks.bench_core import diagnostics as diag
from benchmarks.bench_core.baseline import LegacyRuntime, LegacyRuntimeConfig
from benchmarks.bench_core.diagnostics import RuntimeDiagnostics
from benchmarks.bench_core.ember_harness import (
    EmberBenchServer,
    find_free_port,
    find_or_build_server_binary,
)
from benchmarks.bench_core.workload import WorkloadConfig, build_coordinator

BASELINE_ARM = "Pure Python"
HYBRID_ARM = "Hybrid (Python+EMBER)"

# The distance-to-front value both arms report. Any positive number works;
# what matters is that it is the same one, because
# WebotsExecutorAgent._get_object issues `move_forward(distance)` twice with
# it and a differing value would mean the two arms did different work.
BENCH_DISTANCE_M = 1.25


@dataclass
class RigCapabilities:
    """What a rig can actually do on this machine, so callers can skip a
    measurement rather than report a fabricated one."""

    has_ember: bool = False
    can_read_remote_stats: bool = False
    notes: list[str] = field(default_factory=list)


class Rig:
    """Common surface. Subclasses fill in `start`/`stop` and the client."""

    arm: str = ""

    def __init__(self, workload: WorkloadConfig | None = None):
        self.workload = workload or WorkloadConfig()
        self.diagnostics = RuntimeDiagnostics(self.arm)
        self.capabilities = RigCapabilities()
        self.client: Any = None
        self.driver: EmberRobotDriver | None = None
        self.coordinator: Coordinator | None = None

        # Arrival timestamps for telemetry frames, recorded only while
        # explicitly enabled. Always-on recording would grow an unbounded list
        # during the throughput suites and turn a telemetry benchmark into a
        # measurement of list reallocation.
        self._arrival_lock = threading.Lock()
        self._arrivals: list[int] = []
        self._recording_arrivals = False

    def _note_telemetry(self, frame: TelemetryFrame) -> None:
        """Called on whichever thread delivered the frame (the asyncio loop
        thread in both arms). Kept to a timestamp append under a short lock:
        anything heavier here would be charged to the telemetry path it is
        supposed to be observing."""
        if not self._recording_arrivals:
            return
        now = time.perf_counter_ns()
        with self._arrival_lock:
            self._arrivals.append(now)

    def start_recording_telemetry(self) -> None:
        with self._arrival_lock:
            self._arrivals.clear()
        self._recording_arrivals = True

    def stop_recording_telemetry(self) -> list[int]:
        self._recording_arrivals = False
        with self._arrival_lock:
            return list(self._arrivals)

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __enter__(self) -> "Rig":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ---- shared operations --------------------------------------------

    def send_command(self, op: CommandOp, param: float = 0.0, request_id: str = "") -> AckResult:
        return self.client.send_command(op, param, request_id)

    def latest_telemetry(self, subsystem: TelemetrySubsystem) -> TelemetryFrame | None:
        return self.client.get_latest_telemetry(subsystem)

    def process_pids(self) -> list[tuple[str, int]]:
        """Every OS process this arm's work happens in, for resource sampling.

        The hybrid arm returns two. That is the point: a memory comparison
        that counted only the Python interpreter would show offloading as a
        pure saving, when part of it is relocation.
        """
        import os

        return [(f"{self.arm} / python", os.getpid())]

    def remote_stats(self) -> dict[str, float | int]:
        """Counters from the runtime process, empty for arms that have none."""
        return {}

    def refresh_diagnostics(self) -> None:
        remote = self.remote_stats()
        if remote:
            self.diagnostics.set_remote_counters({k: int(v) for k, v in remote.items()})

    def set_actuation_us(self, value: int) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def start_telemetry_stream(self, hz: float) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def stop_telemetry_stream(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def wait_for_telemetry(self, timeout_s: float = 5.0) -> bool:
        """Blocks until a Motion sample has arrived.

        The executor's `_get_object` multiplies by `get_distance_to_front()`
        and bails out with `False` when it reads 0.0, so a workload started
        before the first telemetry sample lands would silently measure the
        failure path — which is both faster and not the thing under test.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.latest_telemetry(TelemetrySubsystem.MOTION)
            if frame is not None and frame.value_a > 0.0:
                return True
            time.sleep(0.005)
        return False


class BaselineRig(Rig):
    """Pure-Python arm: `LegacyRuntime` underneath the same driver and
    pipeline the hybrid arm uses."""

    arm = BASELINE_ARM

    def __init__(
        self,
        workload: WorkloadConfig | None = None,
        codec_name: str = "json",
        transport: str = "in_process",
        actuation_us: int = 0,
        telemetry_hz: float = 0.0,
    ):
        self.arm = f"{BASELINE_ARM} [{codec_name}/{transport}]"
        super().__init__(workload)
        self.config = LegacyRuntimeConfig(
            codec_name=codec_name,
            transport=transport,
            actuation_us=actuation_us,
            telemetry_hz=telemetry_hz,
        )
        self.runtime: LegacyRuntime | None = None

    def start(self) -> None:
        self.runtime = LegacyRuntime(self.config, self.diagnostics)
        self.runtime.start()
        if not self.runtime.wait_connected(10.0):
            raise RuntimeError("LegacyRuntime failed to become ready")

        self.client = self.runtime
        self.runtime.set_telemetry_callback(self._note_telemetry)
        # The same EmberRobotDriver class as the hybrid arm: it only calls
        # send_command/get_latest_telemetry, both of which LegacyRuntime
        # implements with identical signatures and identical semantics. Using
        # one driver for both arms removes a whole class of "the two rigs
        # diverged" bug from the comparison.
        self.driver = EmberRobotDriver(self.client, camera_width_px=self.workload.camera_width_px)
        self.coordinator = build_coordinator(self.driver, self.workload)

        # Seed a last-known distance so the executor's servo path behaves the
        # same here as it does against EMBER's periodic Motion stream.
        self.runtime.seed_telemetry(TelemetrySubsystem.MOTION, BENCH_DISTANCE_M)
        self.capabilities = RigCapabilities(
            has_ember=False,
            can_read_remote_stats=False,
            notes=[f"codec={self.config.codec_name}", f"transport={self.config.transport}"],
        )

    def stop(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()
            self.runtime = None

    def set_actuation_us(self, value: int) -> None:
        # Read by the worker coroutines on their next iteration; no restart
        # needed, matching the hybrid rig's ACTUATION_US control.
        self.config.actuation_us = value
        if self.runtime is not None:
            self.runtime.config.actuation_us = value

    def start_telemetry_stream(self, hz: float) -> None:
        if self.runtime is None:
            return
        # LegacyRuntime builds its telemetry task at startup from the config,
        # so changing the rate means a restart. Cheap here (no subprocess) and
        # it keeps the task's absolute-deadline pacing correct, which
        # retargeting a running task would not.
        self.stop()
        self.config.telemetry_hz = hz
        self.start()

    def stop_telemetry_stream(self) -> None:
        if self.config.telemetry_hz <= 0:
            return
        self.stop()
        self.config.telemetry_hz = 0.0
        self.start()

    def suspend_heartbeat(self) -> None:
        if self.runtime is not None:
            self.runtime.suspend_heartbeat()

    @property
    def watchdog_stop_count(self) -> int:
        return self.runtime.watchdog_stop_count if self.runtime is not None else 0


class HybridRig(Rig):
    """Hybrid arm: the real `EmberBridgeClient` against the real compiled
    EMBER runtime, in its own OS process, over a real loopback socket."""

    arm = HYBRID_ARM

    def __init__(
        self,
        workload: WorkloadConfig | None = None,
        binary_path: Path | None = None,
        actuation_us: int = 0,
        watchdog_ms: int = 250,
        runtime_hz: int = 200,
        probe_period_us: int = 5000,
        max_telemetry_buffer: int = 256,
        timer_resolution_ms: int = 1,
        send_timeout_ms: int = 50,
        command_timeout_s: float = 3.0,
        log_commands: bool = False,
    ):
        super().__init__(workload)
        self.binary_path = binary_path
        self.actuation_us = actuation_us
        self.watchdog_ms = watchdog_ms
        self.runtime_hz = runtime_hz
        self.probe_period_us = probe_period_us
        self.max_telemetry_buffer = max_telemetry_buffer
        self.timer_resolution_ms = timer_resolution_ms
        self.send_timeout_ms = send_timeout_ms
        self.command_timeout_s = command_timeout_s
        self.log_commands = log_commands
        self.server: EmberBenchServer | None = None
        self.relay: EmberTelemetryRelay | None = None

    def start(self) -> None:
        binary = self.binary_path or find_or_build_server_binary()
        if binary is None:
            raise RuntimeError(
                "No EMBER benchmark server binary: install a C++20 toolchain (g++), "
                "build edge/'s ember_bench_harness_server target, or set "
                "EMBER_BENCH_SERVER_BIN to a prebuilt binary."
            )

        port = find_free_port()
        self.server = EmberBenchServer(
            binary,
            port=port,
            watchdog_ms=self.watchdog_ms,
            actuation_us=self.actuation_us,
            runtime_hz=self.runtime_hz,
            probe_period_us=self.probe_period_us,
            send_timeout_ms=self.send_timeout_ms,
            max_telemetry_buffer=self.max_telemetry_buffer,
            timer_resolution_ms=self.timer_resolution_ms,
            log_commands=self.log_commands,
        )
        if not self.server.wait_ready(timeout=20.0):
            lines = "\n".join(self.server.all_lines())
            self.server.stop()
            self.server = None
            raise RuntimeError(f"EMBER benchmark server never printed READY:\n{lines}")

        self.client = EmberBridgeClient(
            host="127.0.0.1", port=port, command_timeout_s=self.command_timeout_s
        )
        self.client.start()
        if not self.client.wait_connected(10.0):
            self.stop()
            raise RuntimeError("EmberBridgeClient failed to connect to the EMBER benchmark server")

        self.driver = EmberRobotDriver(self.client, camera_width_px=self.workload.camera_width_px)
        self.coordinator = build_coordinator(self.driver, self.workload)

        # The relay is wired in after the Coordinator exists, for the circular
        # -dependency reason EmberBridgeClient.set_telemetry_callback
        # documents. Wiring it at all (rather than leaving telemetry
        # uncollected) keeps its dispatch cost inside the measured system,
        # since a real deployment pays it.
        self.relay = EmberTelemetryRelay(self.coordinator, planner_agent_name="Planner")
        self.client.set_telemetry_callback(self._on_telemetry)

        self.capabilities = RigCapabilities(
            has_ember=True,
            can_read_remote_stats=True,
            notes=[
                f"binary={binary.name}",
                f"port={port}",
                f"runtime_hz={self.runtime_hz}",
                f"timer_resolution_ms={self.timer_resolution_ms}",
            ],
        )

    def _on_telemetry(self, frame: TelemetryFrame) -> None:
        self._note_telemetry(frame)
        self.diagnostics.counter(diag.TELEMETRY_RECEIVED).increment()
        self.diagnostics.rate(diag.TELEMETRY_LOOP).mark()
        if self.relay is not None:
            self.relay.on_telemetry(frame)

    def stop(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None
        if self.server is not None:
            self.server.stop()
            self.server = None

    def process_pids(self) -> list[tuple[str, int]]:
        import os

        pids = [(f"{self.arm} / python", os.getpid())]
        if self.server is not None and self.server.is_alive():
            pids.append((f"{self.arm} / ember", self.server.pid))
        return pids

    def remote_stats(self) -> dict[str, float | int]:
        return self.server.stats() if self.server is not None else {}

    def scheduler_stats(self) -> dict[str, float | int]:
        return self.server.scheduler_stats() if self.server is not None else {}

    def reset_remote_stats(self) -> None:
        if self.server is not None:
            self.server.reset_stats()

    def set_actuation_us(self, value: int) -> None:
        if self.server is not None:
            self.server.set_actuation_us(value)

    def start_telemetry_stream(self, hz: float) -> None:
        if self.server is not None:
            self.server.start_telemetry(hz, BENCH_DISTANCE_M)

    def stop_telemetry_stream(self) -> None:
        if self.server is not None:
            self.server.stop_telemetry()

    @property
    def watchdog_stop_count(self) -> int:
        stats = self.remote_stats()
        return int(stats.get("watchdog_stops", 0))


def ember_available() -> tuple[bool, str]:
    """Whether the hybrid arm can run here, and why not if it can't.

    Called once before any benchmarking so a missing toolchain produces one
    clear message up front rather than an exception partway through a long
    run that has already printed half a report.
    """
    try:
        binary = find_or_build_server_binary()
    except RuntimeError as exc:
        return False, f"EMBER build failed: {exc}"
    if binary is None:
        return False, (
            "no C++ toolchain (g++) and no prebuilt ember_bench_harness_server found; "
            "set EMBER_BENCH_SERVER_BIN or build edge/'s CMake target"
        )
    return True, str(binary)
