from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.features.audio import ContinuousAudioStream
import json

speech_router = APIRouter()

@speech_router.get("/stream-speech")
async def stream_speech():
    def audio_generator():
        streamer = ContinuousAudioStream(chunk_duration=3)
        
        # Iterates infinitely as long as the connection is open
        for transcript in streamer.stream_and_transcribe():
            # Format as Server-Sent Events (SSE) so frontend can read it easily
            yield f"data: {json.dumps({'transcript': transcript})}\n\n"

    return StreamingResponse(audio_generator(), media_type="text/event-stream")