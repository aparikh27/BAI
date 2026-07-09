import json
import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.features.detect import DetectorService
from backend.features.__pycache__.audio import Audio

detect_router = APIRouter()
detector_service = DetectorService()
audio_recorder = Audio(duration=1.5)

class DetectRequest(BaseModel):
    source: str | int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

@detect_router.post('/detect')
async def detect(request: DetectRequest):
    started = detector_service.start(
        source=request.source,
        confidence=request.confidence,
    )

    if not started:
        return {"status": "Detection already running"}

    audio_recorder.start_listening()
    return {"status": "Detection started"}


@detect_router.post('/stopDetect')
async def stopDetect():
    stopped = detector_service.stop()

    if not stopped:
        return {"status": "Detection is not running"}

    audio_recorder.stop_listening()
    return {"status": "Detection stopped"}

@detect_router.get('/video-feed')
def video_feed():
    """Streams the MJPEG video feed to the frontend"""
    if not detector_service.running:
        raise HTTPException(status_code=400, detail="Detection is not running")

    def frame_generator():
        while detector_service.running:
            frame_bytes = detector_service.get_latest_frame()
            
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            time.sleep(0.03)

    return StreamingResponse(
        frame_generator(), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@detect_router.get('/visible-objects')
def visible_objects():
    """Returns the list of currently visible objects in the world memory"""
    if not detector_service.running:
        raise HTTPException(status_code=400, detail="Detection is not running")

    visible_objects = detector_service.world.get_visible_objects()

    return {"visible_objects": [asdict(obj) for obj in visible_objects]}


@detect_router.get('/stream-speech')
def stream_speech():
    """Streams microphone transcriptions to the frontend as server-sent events."""

    def generate():
        while audio_recorder.listening:
            try:
                transcript = audio_recorder.transcribe_microphone().strip()
                if transcript:
                    payload = json.dumps({"transcript": transcript})
                    yield f"data: {payload}\n\n"
            except Exception as exc:
                print(f"Speech streaming error: {exc}")
                payload = json.dumps({"transcript": "", "error": str(exc)})
                yield f"data: {payload}\n\n"

            time.sleep(0.2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )