"""
speech_api.py — Voice command SSE endpoint
─────────────────────────────────────────────
Streams dashboard telemetry over SSE and routes transcription / planning /
execution through the central ``Coordinator`` message bus.  No direct agent
coupling.

Two producers feed the stream:

* the live microphone loop (best-effort — a machine with no capture device
  still gets a working dashboard), and
* ``POST /inject-voice``, which replays a recorded clip through the exact same
  Audio → Planner → Executor path.  Plan steps are dispatched one at a time so
  the dashboard can report progress and a final completion state rather than
  waiting on a single opaque call.
"""

import json
import os
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.messaging import Message, MessageStatus, MessageType

from backend.features.audio import ContinuousAudioStream
from backend.features.events import dashboard_events

speech_router = APIRouter()

_coordinator = None
_coordinator_lock = threading.Lock()

# Set while a recorded clip is being replayed, so ambient microphone chatter
# cannot overwrite the transcript mid-run.
_injection_active = threading.Event()
_injection_lock = threading.Lock()

DEFAULT_DEMO_AUDIO = os.path.join(os.path.dirname(__file__), "..", "Demo.m4a")


def set_coordinator(coordinator) -> None:
    """Called once from ``main.py`` lifespan to wire the message bus."""
    global _coordinator
    with _coordinator_lock:
        _coordinator = coordinator


def _get_coordinator():
    with _coordinator_lock:
        return _coordinator


def _dispatch(coordinator, receiver: str, action: str, payload: dict) -> Message:
    """Send a single request through ``Coordinator.dispatch``."""
    return coordinator.dispatch(
        Message(
            sender="speech_api",
            receiver=receiver,
            action=action,
            payload=dict(payload),
            status=MessageStatus.PENDING,
            message_type=MessageType.REQUEST,
        )
    )


def _merge_payload(base: dict, response: Message) -> dict:
    """Accumulate context from an agent response into the running payload."""
    merged = dict(base)
    if isinstance(response.payload, dict):
        merged.update(response.payload)
    return merged


def _format_command(plan) -> str:
    if not plan:
        return "No robot command detected."
    return json.dumps(plan)


def _describe_step(step: dict) -> str:
    action = str(step.get("action", "?")).replace("_", " ")
    target = step.get("target")
    return f"{action} → {target}" if target else action


# ══════════════════════════════════════════════════════════════════════════
# Recorded-clip replay
# ══════════════════════════════════════════════════════════════════════════

class InjectVoiceRequest(BaseModel):
    audio_path: str | None = None


def _run_voice_pipeline(coordinator, audio_path: str) -> None:
    """Runs one clip through Audio → Planner → Executor, narrating to the hub."""
    publish = dashboard_events.publish
    try:
        publish({
            "stage": "transcribing",
            "transcript": "Transcribing voice command...",
            "task_state": "running",
        })

        response = _dispatch(
            coordinator,
            receiver="Audio",
            action="transcribe",
            payload={"audio_path": audio_path},
        )

        if response.status != MessageStatus.SUCCESS:
            publish({
                "stage": "error",
                "task_state": "failed",
                "message": f"Transcription failed: {response.error}",
            })
            return

        text = (response.payload.get("text") or "").strip()
        if not text:
            publish({
                "stage": "error",
                "task_state": "failed",
                "message": "Transcription returned no speech.",
            })
            return

        publish({"stage": "transcribed", "transcript": text, "task_state": "running"})

        # ── Planning ──────────────────────────────────────────────────
        publish({"stage": "planning", "command": "Planning...", "task_state": "running"})

        planner_response = _dispatch(
            coordinator,
            receiver="Planner",
            action="create_plan",
            payload=_merge_payload({"text": text}, response),
        )

        plan = planner_response.payload.get("plan", []) if planner_response.payload else []
        publish({"stage": "planned", "command": _format_command(plan), "plan": plan})

        if planner_response.status != MessageStatus.SUCCESS or not plan:
            publish({
                "stage": "error",
                "task_state": "failed",
                "message": planner_response.error or "Planner produced an empty plan.",
            })
            return

        # ── Execution, one step at a time so progress is observable ───
        total = len(plan)
        for index, step in enumerate(plan):
            if not isinstance(step, dict):
                publish({
                    "stage": "error",
                    "task_state": "failed",
                    "message": f"Step {index + 1} is malformed: {step!r}",
                })
                return

            publish({
                "stage": "executing",
                "task_state": "running",
                "step_index": index + 1,
                "step_total": total,
                "step_label": _describe_step(step),
                "step_state": "running",
            })

            try:
                executor_response = _dispatch(
                    coordinator,
                    receiver="Executor",
                    action="execute",
                    payload={"step": step},
                )
            except Exception as exc:
                publish({
                    "stage": "error",
                    "task_state": "failed",
                    "message": f"Executor dispatch error: {exc}",
                })
                return

            if executor_response.status != MessageStatus.SUCCESS:
                publish({
                    "stage": "error",
                    "task_state": "failed",
                    "step_index": index + 1,
                    "step_total": total,
                    "step_label": _describe_step(step),
                    "step_state": "failed",
                    "message": executor_response.error or "Execution failed.",
                })
                return

            publish({
                "stage": "executing",
                "task_state": "running",
                "step_index": index + 1,
                "step_total": total,
                "step_label": _describe_step(step),
                "step_state": "done",
            })

        publish({
            "stage": "complete",
            "task_state": "completed",
            "step_total": total,
            "message": f"Task completed — {total} step(s) executed.",
        })

    except Exception as exc:  # pragma: no cover - defensive; keeps the UI honest
        publish({
            "stage": "error",
            "task_state": "failed",
            "message": f"Unexpected pipeline error: {exc}",
        })
    finally:
        # Terminal state has been broadcast to everyone currently watching;
        # don't replay it to whoever connects next.
        dashboard_events.end_run()
        _injection_active.clear()


@speech_router.post("/inject-voice")
async def inject_voice(payload: InjectVoiceRequest | None = None):
    """Replays a recorded clip through the live agent pipeline.

    Returns as soon as the run is accepted; progress arrives over
    ``/stream-speech`` so the dashboard renders it exactly like live speech.
    """
    coordinator = _get_coordinator()
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Coordinator is not initialized. Is Webots connected?",
        )

    requested = (payload.audio_path if payload else None) or DEFAULT_DEMO_AUDIO
    audio_path = os.path.abspath(requested)

    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    with _injection_lock:
        if _injection_active.is_set():
            raise HTTPException(status_code=409, detail="A voice command is already running.")
        _injection_active.set()

    dashboard_events.start_run()
    threading.Thread(
        target=_run_voice_pipeline,
        args=(coordinator, audio_path),
        daemon=True,
    ).start()

    return {"status": "Voice command accepted", "audio_path": audio_path}


# ══════════════════════════════════════════════════════════════════════════
# Live microphone capture (best-effort)
# ══════════════════════════════════════════════════════════════════════════

def _run_microphone_loop(coordinator, stop_event: threading.Event) -> None:
    """Transcribes live mic audio into the hub until ``stop_event`` is set.

    Capture hardware is optional: on a machine with no input device this logs
    once and returns, leaving the SSE stream fully functional for replayed
    clips.
    """
    streamer = ContinuousAudioStream(chunk_duration=3)
    try:
        for audio_block in streamer.stream_audio(stop_event=stop_event):
            if stop_event.is_set():
                break

            # A replayed clip owns the transcript while it runs.
            if _injection_active.is_set():
                continue

            response = _dispatch(
                coordinator,
                receiver="Audio",
                action="transcribe",
                payload={"audio_data": audio_block},
            )

            if response.status != MessageStatus.SUCCESS:
                if response.error:
                    print(f"[SPEECH-API] Transcription error: {response.error}")
                continue

            text = (response.payload.get("text") or "").strip()
            if not text:
                continue

            dashboard_events.publish({"transcript": text, "source": "microphone"})

            planner_response = _dispatch(
                coordinator,
                receiver="Planner",
                action="create_plan",
                payload=_merge_payload({"text": text}, response),
            )
            plan = planner_response.payload.get("plan", []) if planner_response.payload else []
            dashboard_events.publish({"command": _format_command(plan), "plan": plan})

            if planner_response.status != MessageStatus.SUCCESS or not plan:
                continue

            _dispatch(
                coordinator,
                receiver="Executor",
                action="execute",
                payload=_merge_payload({"text": text}, planner_response),
            )
    except Exception as exc:
        # No microphone is a normal condition here, not a dashboard failure.
        print(f"[SPEECH-API] Microphone capture unavailable: {exc}")
        dashboard_events.publish({
            "microphone": "unavailable",
            "message": "Microphone unavailable — replayed commands still work.",
        })


@speech_router.get("/stream-speech")
async def stream_speech(mic: int = 1):
    coordinator = _get_coordinator()
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Coordinator is not initialized. Is Webots connected?",
        )

    def event_generator():
        subscriber = dashboard_events.subscribe()
        stop_event = threading.Event()

        if mic:
            threading.Thread(
                target=_run_microphone_loop,
                args=(coordinator, stop_event),
                daemon=True,
            ).start()

        try:
            # Announce immediately so the dashboard can flip its telemetry
            # indicator without waiting on the first spoken word.
            yield f"data: {json.dumps({'connected': True})}\n\n"

            while True:
                try:
                    event = subscriber.get(timeout=10)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"
        finally:
            stop_event.set()
            dashboard_events.unsubscribe(subscriber)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
