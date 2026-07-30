"""
detect_api.py — Vision streaming & robot command endpoints
───────────────────────────────────────────────────────────
``DetectorService`` handles continuous YOLO perception for the UI.
Robot actions route through ``Coordinator.dispatch`` to the Executor agent.
"""

import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from agents.messaging import Message, MessageStatus, MessageType
from pydantic import BaseModel, Field

from backend.features.audio import Audio
from backend.features.detect import DetectorService
from backend.features.memory import World

detect_router = APIRouter()

_coordinator = None
detector_service: DetectorService | None = None
audio_recorder = Audio(duration=1)


def set_coordinator(coordinator) -> None:
    """Called once from ``main.py`` lifespan to wire the message bus."""
    global _coordinator
    _coordinator = coordinator


def _get_detector_service(app) -> DetectorService:
    """Lazy-init ``DetectorService`` bound to the shared app ``World``."""
    global detector_service

    world = getattr(app.state, "world", None)
    driver = getattr(app.state, "robot_driver", None)

    if detector_service is None:
        detector_service = DetectorService(
            webots_driver=driver,
            world=world if isinstance(world, World) else None,
        )
    else:
        if isinstance(world, World):
            detector_service.world = world
        if driver is not None:
            detector_service.camera.webots_driver = driver
            detector_service.camera.use_webots = True

    return detector_service


def _dispatch_executor(coordinator, command: str, target_id: str) -> Message:
    """Route a single robot action through the Executor agent."""
    return coordinator.dispatch(
        Message(
            sender="detect_api",
            receiver="Executor",
            action="execute",
            payload={
                "action": command,
                "target": target_id,
            },
            status=MessageStatus.PENDING,
            message_type=MessageType.REQUEST,
        )
    )


class DetectRequest(BaseModel):
    source: str | int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RobotCommandRequest(BaseModel):
    command: str
    target_id: str


@detect_router.post("/detect")
async def detect(request: Request, detect_req: DetectRequest):
    coordinator = getattr(request.app.state, "coordinator", None) or _coordinator
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator is not initialized.")

    service = _get_detector_service(request.app)
    started = service.start(
        source=detect_req.source,
        confidence=detect_req.confidence,
    )

    if not started:
        return {"status": "Detection already running"}

    driver = getattr(request.app.state, "robot_driver", None)
    if driver is not None:
        service.camera.webots_driver = driver
        service.camera.use_webots = True
        print("[CAMERA] Wired Webots robot camera to DetectorService")

    audio_recorder.start_listening()
    return {"status": "Detection started"}


@detect_router.post("/execute-robot")
async def execute_robot(request: Request, payload: RobotCommandRequest):
    coordinator = getattr(request.app.state, "coordinator", None) or _coordinator
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator is not initialized.")

    service = _get_detector_service(request.app)
    if not service.running:
        raise HTTPException(
            status_code=400,
            detail="Cannot run robot commands while vision detection is stopped.",
        )

    print(
        f"⚡ API received request: Action='{payload.command}' "
        f"on Object ID={payload.target_id}"
    )

    response = _dispatch_executor(coordinator, payload.command, payload.target_id)

    if response.status != MessageStatus.SUCCESS:
        return {
            "status": "Execution failed",
            "success": False,
            "error": response.error,
        }

    return {"status": "Task successfully executed", "success": True}


@detect_router.post("/stopDetect")
async def stopDetect():
    if detector_service is None or not detector_service.running:
        return {"status": "Detection is not running"}

    detector_service.stop()
    audio_recorder.stop_listening()
    return {"status": "Detection stopped"}


@detect_router.get("/video-feed")
def video_feed(request: Request):
    service = _get_detector_service(request.app)
    if not service.running:
        raise HTTPException(status_code=400, detail="Detection is not running")

    def frame_generator():
        while service.running:
            frame_bytes = service.get_latest_frame()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@detect_router.get("/visible-objects")
def visible_objects(request: Request):
    service = _get_detector_service(request.app)
    if not service.running:
        raise HTTPException(status_code=400, detail="Detection is not running")

    visible = service.world.get_visible_objects()
    return {"visible_objects": [asdict(obj) for obj in visible]}
