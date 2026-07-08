from backend.features.detect import DetectorService
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import time
from fastapi.responses import StreamingResponse
from dataclasses import asdict  # <--- Make sure to import this at the top

detect_router = APIRouter()
detector_service = DetectorService()

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

    return {"status": "Detection started"}

@detect_router.post('/stopDetect')
async def stopDetect():
    stopped = detector_service.stop()

    if not stopped:
        return {"status": "Detection is not running"}

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