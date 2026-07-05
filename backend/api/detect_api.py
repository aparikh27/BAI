from backend.features.detect import DetectorService
from pydantic import BaseModel, Field
from fastapi import APIRouter

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
