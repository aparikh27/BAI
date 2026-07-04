from backend.image_detection.yolo_detector import YOLODetector
from pydantic import BaseModel, Field
from fastapi import APIRouter

detect_router = APIRouter()
detector = YOLODetector()

class DetectRequest(BaseModel):
    source: str | int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

@detect_router.post('/detect')
async def detect(request: DetectRequest):
    results = detector.detect(source=request.source, confidence=request.confidence)
    for result in results:
        print(result.verbose())
    return {"status": "success", "message": f"Finished processing {request.source}"}