from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Webots internal imports (enabled by your PYTHONPATH variable)
from controller import Robot  

# Your custom modular codebase imports
from backend.api.detect_api import detect_router
from backend.api.speech_api import speech_router
from backend.robot_execution.webot import WebotDriver
from backend.robot_execution.execution_logic import RobotExecutor
from backend.features.memory import World

# 1. Create a global dictionary or container to store the executor 
# so your routers/endpoints can access it later.
robot_app_state = {}
world_memory = World()

# 2. Define the Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🤖 FastAPI Lifespan Starting: Connecting to Webots Simulator...")
    try:
        # Initialize Webots interface link
        robot_instance = Robot()
        driver = WebotDriver(robot_instance)
        driver.initialize_devices()
        
        # Instantiate your execution engine bridge
        executor = RobotExecutor(robot_driver=driver, world=world_memory)
        
        # Save it to our state container so endpoints can use it
        robot_app_state["executor"] = executor
        robot_app_state["world"] = world_memory
        print("🎯 Handshake complete! Connected to Webots successfully.")
    except Exception as e:
        print(f"❌ Failed to link with Webots: {e}")
        print("💡 Make sure Webots is open, set to <extern>, and playing!")

    yield
    # Cleanup actions when server shuts down go here
    print("🔌 FastAPI Lifespan Stopping: Disconnecting from Webots.")

# 3. Pass lifespan to FastAPI initialization
app = FastAPI(lifespan=lifespan)

# Store the state directly on the app instance for clean dependency injection
app.state.robot_executor = robot_app_state

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

# Include your existing routers
app.include_router(detect_router, prefix="/api")
app.include_router(speech_router, prefix="/api")