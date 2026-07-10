from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.features.audio import ContinuousAudioStream
import json
from backend.brain.brain_qwen import QWENBRAIN

speech_router = APIRouter()
streamer = None
brain = None


def get_speech_components():
    global streamer, brain
    if streamer is None:
        streamer = ContinuousAudioStream(chunk_duration=3)
    if brain is None:
        brain = QWENBRAIN()
    return streamer, brain


@speech_router.get("/stream-speech")
async def stream_speech():
    streamer_instance, brain_instance = get_speech_components()

    def audio_generator():
        for transcript in streamer_instance.stream_and_transcribe():
            if not transcript.strip():
                continue

            try:
                command = brain_instance.process_task(transcript)
            except Exception as exc:
                print(f"Speech command processing error: {exc}")
                command = ""

            if not command or command == "[]":
                command = "No robot command detected."

            yield f"data: {json.dumps({'transcript': transcript, 'command': command})}\n\n"

    return StreamingResponse(
        audio_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )