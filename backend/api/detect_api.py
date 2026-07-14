import json
import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request  # Added Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.features.detect import DetectorService
from backend.features.audio import Audio

detect_router = APIRouter()
detector_service = DetectorService()
audio_recorder = Audio(duration=1.5)

class DetectRequest(BaseModel):
    source: str | int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

# 1. New Request Schema for your React Frontend to trigger the robot
class RobotCommandRequest(BaseModel):
    command: str  # e.g., "get_object", "put_object"
    target_id: str  # The track ID string from YOLO

@detect_router.post('/detect')
async def detect(request: Request, detect_req: DetectRequest):  # Added FastAPI Request parameter
    started = detector_service.start(
        source=detect_req.source,
        confidence=detect_req.confidence,
    )

    if not started:
        return {"status": "Detection already running"}

    # Link World Memory and Robot Camera! 
    if hasattr(request.app.state, "robot_executor"):
        app_state = request.app.state.robot_executor
        if "executor" in app_state:
            # Override the executor's world reference with this endpoint's live world tracker
            app_state["executor"].world = detector_service.world
            print("[LINK] Connected Robot Executor memory to the active YOLO Detector World!")
        
        # Wire the robot camera to the detector's camera service
        if app_state.get("executor") and hasattr(app_state["executor"], "robot"):
            robot_driver = app_state["executor"].robot
            detector_service.camera.webots_driver = robot_driver
            detector_service.camera.use_webots = True
            print("[CAMERA] Wired Webots robot camera to DetectorService (replacing local webcam)")

    audio_recorder.start_listening()
    return {"status": "Detection started"}


# 3. New API Endpoint for React to control the robot actions
@detect_router.post('/execute-robot')
async def execute_robot(request: Request, payload: RobotCommandRequest):
    """Receives a task command from the React frontend and executes it on the Webots robot."""
    if not detector_service.running:
        raise HTTPException(status_code=400, detail="Cannot run robot commands while vision detection is stopped.")

    # Grab the active executor instance from app lifecycle state
    if not hasattr(request.app.state, "robot_executor") or "executor" not in request.app.state.robot_executor:
        raise HTTPException(status_code=503, detail="Robot driver is not initialized or connected to Webots.")

    executor = request.app.state.robot_executor["executor"]

    print(f"⚡ API received request: Action='{payload.command}' on Object ID={payload.target_id}")
    
    # Run the visual servoing tracking and navigation loops we built
    # (Note: This runs synchronously and blocks until the robot returns to origin)
    success = executor.execute_command(payload.command, payload.target_id)
    
    if not success:
        return {"status": "Execution failed", "success": False}
        
    return {"status": "Task successfully executed", "success": True}


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