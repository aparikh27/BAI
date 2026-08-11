"""
main.py — BAI Robotics Entry Point
────────────────────────────────────
Central wiring point for the BAI project.  Boots the ``agents`` submodule,
instantiates a ``Coordinator`` as the single message bus, registers every
concrete agent, and exposes the coordinator to the FastAPI API layer.
"""

# ── 0. Console encoding (must precede any agent import) ──────────────────
# Agent banners and this module's own logging use emoji and box-drawing
# characters. On Windows the default console/redirect codepage is cp1252, so
# those prints raise UnicodeEncodeError — inside ``add_agent`` that surfaces as
# a fatal startup error and leaves the coordinator uninitialised, which the API
# then reports as a Webots connection failure. Force UTF-8 with replacement so
# logging can never take the application down.
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - exotic stream types
        pass

# ── Standard / third-party ────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Webots is available only inside the simulator controller runtime.  Importing
# the API module must still work for route tests and non-simulator tooling.
# ``load_robot_class`` resolves the *real* Webots API from WEBOTS_HOME; the
# repo-root ``controller.py`` stub would otherwise shadow it and leave the
# coordinator permanently uninitialised.
from backend.webots_runtime import load_robot_class

Robot = load_robot_class()

# ── Agents submodule public API ───────────────────────────────────────────
from agents.master_planner.coordinator import Coordinator
from agents.master_planner.pipeline import PipelineStep
from agents.agent.audio_agent.whisper_audio import WhisperAudioAgent
from agents.agent.vision_agent.yolo_vision import YOLOVisionAgent
from agents.agent.planner_agent.qwen_planner import QwenPlannerAgent
from agents.agent.execution_agent.webot_execution import WebotsExecutorAgent
from agents.agent.memory_agent.memory_engine_agent import MemoryEngineAgent

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

    if Robot is None:
        print("[BAI] Webots controller module is unavailable; API is running without robot hardware.")
        yield
        return

    try:
        # ── Hardware init ─────────────────────────────────────────────
        robot_instance = Robot()
        driver = WebotDriver(robot_instance)
        driver.initialize_devices()

        # ── Coordinator ───────────────────────────────────────────────
        coordinator = Coordinator()

        # ── Register all agents ───────────────────────────────────────
        audio_agent = WhisperAudioAgent(model_size="tiny")
        planner_agent = QwenPlannerAgent(
            model_path="backend/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
        )

        coordinator.add_agent(audio_agent)
        coordinator.add_agent(YOLOVisionAgent(model_path="yolo11n.pt"))
        coordinator.add_agent(planner_agent)
        coordinator.add_agent(WebotsExecutorAgent(
            robot_driver=driver,
            world=world_memory,
        ))
        coordinator.add_agent(MemoryEngineAgent(
            capacity=100,
            db_path="robot_memory.db",
        ))

        # ── Warm the language models ──────────────────────────────────
        # Whisper and the Qwen planner both load lazily on first use, which
        # puts 30-60s of model loading in front of the *first* voice command
        # while the operator waits. Pay that cost during startup instead.
        # Set BAI_SKIP_MODEL_WARMUP=1 to boot faster during development.
        if os.environ.get("BAI_SKIP_MODEL_WARMUP") != "1":
            for label, warm in (
                ("Whisper", audio_agent._get_model),
                ("Qwen planner", planner_agent._load_model),
            ):
                try:
                    print(f"[BAI] Warming {label}...", flush=True)
                    warm()
                except Exception as exc:
                    # A cold model is a slow first command, never a fatal error.
                    print(f"[BAI] Could not warm {label}: {exc}", flush=True)

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
