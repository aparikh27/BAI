"""
main.py — BAI Robotics Entry Point
────────────────────────────────────
Central wiring point for the BAI project.  Boots the ``agents`` submodule,
instantiates a ``Coordinator`` as the single message bus, registers every
concrete agent, and exposes the coordinator to the FastAPI API layer.
"""

# ── 0. Bootstrap the agents submodule (must be first) ────────────────────
import backend.agents_bootstrap  # noqa: F401  — sys.path side-effect only

# ── Standard / third-party ────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Webots internal imports (enabled by your PYTHONPATH variable)
from controller import Robot  

# ── Agents submodule public API ───────────────────────────────────────────
from master_planner.coordinator import Coordinator
from master_planner.pipeline import PipelineStep
from agent.audio_agent.whisper_audio import WhisperAudioAgent
from agent.vision_agent.yolo_vision import YOLOVisionAgent
from agent.planner_agent.qwen_planner import QwenPlannerAgent
from agent.execution_agent.webot_execution import WebotsExecutorAgent
from agent.memory_agent.memory_engine_agent import MemoryEngineAgent

# ── BAI-specific runtime services (NOT superseded by agents) ──────────────
from backend.robot_execution.webot import WebotDriver
from backend.features.memory import World

# ── API routers ───────────────────────────────────────────────────────────
from backend.api.detect_api import detect_router, set_coordinator as set_detect_coordinator
from backend.api.speech_api import speech_router, set_coordinator as set_speech_coordinator


# ══════════════════════════════════════════════════════════════════════════
# Application-level state
# ══════════════════════════════════════════════════════════════════════════
world_memory = World()


# ══════════════════════════════════════════════════════════════════════════
# Lifespan: initialise hardware + wire Coordinator
# ══════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("═" * 60)
    print("[BAI] Lifespan Starting — Connecting to Webots Simulator...")
    print("═" * 60)

    try:
        # ── Hardware init ─────────────────────────────────────────────
        robot_instance = Robot()
        driver = WebotDriver(robot_instance)
        driver.initialize_devices()

        # ── Coordinator ───────────────────────────────────────────────
        coordinator = Coordinator()

        # ── Register all agents ───────────────────────────────────────
        coordinator.add_agent(WhisperAudioAgent(model_size="tiny"))
        coordinator.add_agent(YOLOVisionAgent(model_path="yolo11n.pt"))
        coordinator.add_agent(QwenPlannerAgent(
            model_path="backend/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
        ))
        coordinator.add_agent(WebotsExecutorAgent(
            robot_driver=driver,
            world=world_memory,
        ))
        coordinator.add_agent(MemoryEngineAgent(
            capacity=100,
            db_path="robot_memory.db",
        ))

        # ── Register BAI-specific custom pipelines ──────────────────
        # Speech-only path (no vision frame): Planner → Executor
        coordinator.pipeline.create_custom_pipeline(
            "speech_to_action",
            [
                PipelineStep(receiver="Planner", action="create_plan"),
                PipelineStep(receiver="Executor", action="execute"),
            ],
        )

        # ── Expose coordinator to API layer ───────────────────────────
        app.state.coordinator = coordinator
        # Keep driver/world accessible for DetectorService camera wiring
        app.state.robot_driver = driver
        app.state.world = world_memory

        # Wire coordinator into both API routers
        set_speech_coordinator(coordinator)
        set_detect_coordinator(coordinator)

        print("═" * 60)
        print("[BAI] ✅ Coordinator online — all agents registered")
        for name in coordinator.all_agents:
            print(f"       ├─ {name}")
        print("[BAI] ✅ Handshake complete! Connected to Webots.")
        print("═" * 60)

    except Exception as e:
        print(f"[ERROR] Failed to link with Webots: {e}")
        print("[TIP] Make sure Webots is open, set to <extern>, and playing!")

    yield

    print("[DISCONNECT] FastAPI Lifespan Stopping: Disconnecting from Webots.")


# ══════════════════════════════════════════════════════════════════════════
# FastAPI application
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(lifespan=lifespan)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect_router, prefix="/api")
app.include_router(speech_router, prefix="/api")