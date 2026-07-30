# BAI — AgentCore Robotics Framework

> A multi-agent orchestration and reinforcement-learning framework for robots that need to perceive, reason, remember, and act.

BAI (Behavioral Autonomy Infrastructure) brings together AgentCore coordination, YOLO perception, speech and planning agents, a persistent world model, and a Gymnasium-compatible Webots environment. It is a practical foundation for autonomous-robot experiments, not a single hard-wired demo.

## Motivation

Most robots today are highly capable, but they are still limited by how much we have explicitly programmed them to do. Traditional robotics systems often follow a fixed pipeline:

Sensor → Perception → Controller → Action

This approach works well for controlled environments, but real-world environments are unpredictable. A robot operating in a home, hospital, warehouse, or disaster zone must handle new situations, incomplete information, and changing goals without requiring every possible scenario to be manually programmed.

BAI explores a more adaptive approach by combining perception, reasoning, memory, and learning into a unified architecture:

Perception → World Model → Memory → Reasoning → Learning → Action

The goal of BAI is to move toward robots that do more than simply execute predefined commands. By allowing robots to build an understanding of their environment, learn from experience, and make decisions based on context, we can create systems that are more flexible, reliable, and useful in the real world.

This type of adaptability has the potential to enable more capable robotic assistants, improve automation in industries facing labor shortages, support healthcare and elder care, and help robots operate safely in complex environments where human intervention is limited.

## System flow

```text
                      AgentCore Coordinator
              (message bus, pipelines, agent routing)
                    /             |             \
          Vision / YOLO       Planner / LLM    Memory Agent
                    \             |             /
                     +------> World Model <----+
                                  |
                         WebotsBridge
                    (real driver or mock driver)
                                  |
        +-------------------------+-------------------------+
        |                                                   |
     BAIEnv <--- ObservationBuilder                  RewardEngine
        |          (Gym observation)                 (progress, goal,
        +--------------------> PPO <----------------- collision rewards)
                         Stable-Baselines3
```

## Feature

- PPO navigation training through Stable-Baselines3 and Gymnasium.
- Webots integration with safe fallback/mock behavior so RL validation can run without a simulator controller.
- A composable `ObservationBuilder` and custom `RewardEngine` for navigation experiments.
- AgentCore message routing for vision, audio, planning, execution, and durable memory agents.
- YOLO/OpenCV perception, Whisper speech transcription, and a FastAPI API surface.
- Installable Python packages: `agents`, `backend`, `rl`, and `MemoryEngine` work after `pip install -e .`; no `PYTHONPATH` setup is required.

## Demo

BAI currently supports:

- Real-time object detection using YOLO
- Voice commands through Whisper speech recognition
- LLM-based task planning
- Persistent world memory
- Autonomous navigation training through PPO
- Webots robot simulation
- React telemetry dashboard

Example task:

User:
> "Find the bottle"

Pipeline:

Voice → Planner → Vision → Memory → PPO Navigation → Robot Execution

## Quickstart

Prerequisites: Python 3.10–3.13 and Git. Webots is optional for the dry run; when it is not available, BAI uses placeholder driver/world objects.

1. Clone the repository and create an isolated environment.

   ```bash
   git clone https://github.com/aparikh27/BAI.git
   cd BAI
   python -m venv venv
   ```

2. Activate it.

   ```bash
   # macOS/Linux
   source venv/bin/activate

   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. Install BAI in editable mode (recommended), or install the runtime/development requirements.

   ```bash
   pip install --upgrade pip
   pip install -e .
   # Alternative, including pytest:
   # pip install -r requirements.txt
   ```

### Reinforcement Learning
1. Validate the RL environment 

   ```bash
   python -m rl.train --dry-run
   ```

2. Start PPO training.

   ```bash
   python -m rl.train --timesteps 100000
   ```

3. Monitor training.

   ```bash
   tensorboard --logdir=logs/ppo_navigation
   ```

4. Run the unit and integration tests.

   ```bash
   pytest
   ```

### Live Control

BAI is a **fully functional, interactive voice- and vision-driven robotic personal assistant**. 

You can interact with the robot in real time through spoken commands while observing its perception, reasoning, and physical execution through a live Webots simulation and custom React telemetry dashboard.

```text
 ┌────────────────┐       Spoken Voice      ┌─────────────────────────┐
 │   User Voice   ├────────────────────────►│  Whisper Audio Agent    │
 └────────────────┘                         └────────────┬────────────┘
                                                         │ Text Command
                                                         ▼
 ┌────────────────┐      Telemetry / Video  ┌─────────────────────────┐
 │ React Dashboard│◄────────────────────────┤  AgentCore Coordinator  │
 └────────────────┘                         └────────────┬────────────┘
                                                         │ Action / Intent
                                                         ▼
 ┌────────────────┐       Camera Stream     ┌─────────────────────────┐
 │ Webots 3D World│◄────────────────────────┤ Vision (YOLO) & Planner │
 └────────────────┘                         └─────────────────────────┘
 ```

## How to Run
### 1. Start the FastAPI Backend
From the project root (with your virtual environment activated):


`python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
### 2. Start the frontend
In a new terminal window
1. `cd frontend`
2. `npm install`
3. `npm run dev`

## Repository layout

```text
agents/          AgentCore coordinator, messaging, and specialist agents
backend/         FastAPI routes, perception services, world model, Webots driver
rl/              BAIEnv, simulator bridge, observations, rewards, PPO scripts
MemoryEngine/    Standalone short- and long-term memory implementation
frontend/        React dashboard
```

## Running with Webots

The RL entry point attempts to import Webots' `controller.Robot` at runtime. Launch it from a configured Webots controller for real simulation interaction. Outside Webots, `python -m rl.train --dry-run` deliberately uses the fallback path, making package and environment validation portable.

## Development notes

- Generated checkpoints belong under `models/` and training telemetry under `logs/`; both are excluded from Git.
- Local model weights, simulator installations, and credentials should remain outside commits. Use `.env` for local secrets.
- The `agents` and `MemoryEngine` directories are Git submodules. After cloning, initialize them with:

  ```bash
  git submodule update --init --recursive
  ```

## Tech Stack

### AI
- PyTorch
- YOLO
- Whisper
- Stable-Baselines3 PPO
- OpenAI-compatible LLM APIs

### Robotics
- Webots
- Gymnasium
- ROS2 (planned)

### Backend
- Python
- FastAPI
- WebSockets

### Frontend
- React
- TypeScript
- TailwindCSS


## License

This project is released under the terms of the [MIT License](LICENSE).
