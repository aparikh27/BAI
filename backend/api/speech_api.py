"""
speech_api.py — Voice command SSE endpoint
─────────────────────────────────────────────
Captures microphone audio, routes transcription / planning / execution
through the central ``Coordinator`` message bus.  No direct agent coupling.
"""

import json
import queue
import threading

import backend.agents_bootstrap  # noqa: F401 — agents submodule on sys.path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from messaging import Message, MessageStatus, MessageType

from backend.features.audio import ContinuousAudioStream

speech_router = APIRouter()

_coordinator = None
_coordinator_lock = threading.Lock()


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


@speech_router.get("/stream-speech")
async def stream_speech():
    coordinator = _get_coordinator()
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Coordinator is not initialized. Is Webots connected?",
        )

    streamer = ContinuousAudioStream(chunk_duration=3)

    def audio_generator():
        stop_event = threading.Event()
        events: queue.Queue = queue.Queue()
        command_queue: queue.Queue = queue.Queue()
        sentinel = object()
        transcription_complete = object()
        command_complete = object()

        def transcribe():
            try:
                for audio_block in streamer.stream_audio(stop_event=stop_event):
                    if stop_event.is_set():
                        break

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

                    text = response.payload.get("text", "").strip()
                    if text:
                        events.put({"transcript": text})
                        command_queue.put(_merge_payload({}, response))
            except Exception as exc:
                print(f"[SPEECH-API] Speech capture error: {exc}")
            finally:
                command_queue.put(sentinel)
                events.put(transcription_complete)

        def process_commands():
            try:
                while not stop_event.is_set():
                    try:
                        payload = command_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if payload is sentinel:
                        break

                    try:
                        planner_response = _dispatch(
                            coordinator,
                            receiver="Planner",
                            action="create_plan",
                            payload=payload,
                        )
                    except Exception as exc:
                        print(f"[SPEECH-API] Planner dispatch error: {exc}")
                        continue

                    plan = planner_response.payload.get("plan", [])
                    events.put({"command": _format_command(plan)})

                    if (
                        planner_response.status != MessageStatus.SUCCESS
                        or not plan
                    ):
                        if planner_response.error:
                            print(f"[SPEECH-API] Planner error: {planner_response.error}")
                        continue

                    executor_payload = _merge_payload(payload, planner_response)
                    try:
                        executor_response = _dispatch(
                            coordinator,
                            receiver="Executor",
                            action="execute",
                            payload=executor_payload,
                        )
                    except Exception as exc:
                        print(f"[SPEECH-API] Executor dispatch error: {exc}")
                        continue

                    if executor_response.status != MessageStatus.SUCCESS:
                        print(
                            f"[SPEECH-API] Execution error: {executor_response.error}"
                        )
            finally:
                events.put(command_complete)

        threading.Thread(target=transcribe, daemon=True).start()
        threading.Thread(target=process_commands, daemon=True).start()

        transcription_done = False
        command_done = False
        try:
            while not (transcription_done and command_done):
                try:
                    event = events.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue

                if event is transcription_complete:
                    transcription_done = True
                elif event is command_complete:
                    command_done = True
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        audio_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
