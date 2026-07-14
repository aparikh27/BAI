from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.features.audio import ContinuousAudioStream
import json
import queue
import threading
from backend.brain.brain_qwen import QWENBRAIN
from backend.robot_execution.execution_logic import RobotExecutor

speech_router = APIRouter()
brain = None
brain_lock = threading.Lock()
robot_executor = None
robot_executor_lock = threading.Lock()


def get_speech_components():
    global brain
    # Capture state belongs to a single browser connection. Reusing an old
    # InputStream causes reconnects to compete for the same microphone.
    if brain is None:
        brain = QWENBRAIN()
    return ContinuousAudioStream(chunk_duration=3), brain


def set_robot_executor(executor: RobotExecutor | None):
    global robot_executor
    with robot_executor_lock:
        robot_executor = executor


def _dispatch_robot_command(command: str):
    if not command or command == "[]" or command == "No robot command detected.":
        print(f"[SPEECH-API] Skipping empty command: {repr(command)}")
        return

    print(f"[SPEECH-API] Attempting to dispatch robot command: {repr(command)}")
    
    try:
        plan = json.loads(command)
    except json.JSONDecodeError:
        print(f"[SPEECH-API] Failed to parse command as JSON: {repr(command)}")
        return

    if not isinstance(plan, list):
        print(f"[SPEECH-API] Command is not a list, got: {type(plan)}")
        return

    with robot_executor_lock:
        executor = robot_executor

    if executor is None:
        print("[SPEECH-API] ERROR: Robot executor is None! Not wired to main.py lifespan.")
        return

    print(f"[SPEECH-API] Executor available. Processing {len(plan)} command steps...")

    for step in plan:
        if not isinstance(step, dict):
            print(f"[SPEECH-API] Skipping non-dict step: {step}")
            continue

        action = step.get("action") or step.get("command")
        target = step.get("target") or step.get("target_item")
        print(f"[SPEECH-API] Executing step: action='{action}', target='{target}'")
        
        if action:
            try:
                executor.execute_command(action, target)
                print(f"[SPEECH-API] Step completed successfully: {action}")
            except Exception as exc:
                print(f"[SPEECH-API] Robot execution error: {exc}")


@speech_router.get("/stream-speech")
async def stream_speech():
    streamer_instance, brain_instance = get_speech_components()

    def audio_generator():
        stop_event = threading.Event()
        events = queue.Queue()
        command_queue = queue.Queue()
        sentinel = object()
        transcription_complete = object()
        command_complete = object()

        def transcribe():
            try:
                for transcript in streamer_instance.stream_and_transcribe(stop_event=stop_event):
                    if stop_event.is_set():
                        break
                    if transcript.strip():
                        # Transcription is sent immediately; Qwen inference must
                        # never block the microphone capture loop.
                        events.put({"transcript": transcript})
                        command_queue.put(transcript)
            except Exception as exc:
                print(f"Speech transcription error: {exc}")
            finally:
                command_queue.put(sentinel)
                events.put(transcription_complete)

        def process_commands():
            try:
                while not stop_event.is_set():
                    try:
                        transcript = command_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if transcript is sentinel:
                        break

                    try:
                        # A reconnect can overlap an in-flight request. Llama's
                        # model instance is shared, so only inference is
                        # serialized; microphone capture remains continuous.
                        with brain_lock:
                            command = brain_instance.process_task(transcript)
                    except Exception as exc:
                        print(f"Speech command processing error: {exc}")
                        command = ""

                    if not command or command == "[]":
                        command = "No robot command detected."
                    events.put({"command": command})
                    _dispatch_robot_command(command)
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
                    # Keep the SSE connection alive during silence.
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
            "X-Accel-Buffering": "no"
        }
    )
