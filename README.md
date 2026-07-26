# BAI: Behavioral Autonomy Infrastructure

<p align="center">
  <b>An autonomous robotics platform powered by multi-agent AI, computer vision, speech recognition, memory, and intelligent task planning.</b>
</p>

---

## Overview

BAI (Behavioral Autonomy Infrastructure) is an AI-powered robotics platform designed to enable autonomous robots to perceive, reason, remember, and act within their environment.

Unlike traditional robotics systems that rely on tightly coupled modules, BAI is built on top of the **ATLAS Agents Framework**, a modular multi-agent architecture where specialized agents communicate through a standardized messaging protocol.

The project combines state-of-the-art AI models with robotics software to create an extensible foundation for intelligent autonomous systems.

---

# Features

- 🎥 Real-time object detection and tracking using YOLO
- 🎙️ Continuous speech recognition using Whisper
- 🧠 LLM-powered task planning
- 💾 Persistent short-term and long-term memory engine
- 🤖 Autonomous robot command execution
- 🌍 Persistent world model for environment understanding
- 🔄 Modular multi-agent architecture
- 📊 Web dashboard for monitoring perception and planning
- 🛰️ Simulation support (Isaac Sim / Webots)

---

# System Architecture

```text
                    User

                      │

            Speech / Text Input

                      │

                Audio Agent
                 (Whisper)

                      │

                      ▼

               Planner Agent
                 (LLM)

                      │

              Task Pipeline

                      │

                 Coordinator
         (ATLAS Agents Framework)

      ┌──────────┼───────────┬───────────┐
      │          │           │           │
      ▼          ▼           ▼           ▼

 Vision      Memory     Execution    Future Agents
  Agent        Agent        Agent

      │          │           │
      └──────────┼───────────┘
                 │
                 ▼

             World Model

                 │
                 ▼

        Robot / Simulator
```

---

# Tech Stack

## AI / Machine Learning

- Python
- PyTorch
- Ultralytics YOLO11
- Faster-Whisper
- Qwen (Planner LLM)
- OpenAI-compatible APIs

---

## Multi-Agent Framework

BAI is built on top of the **ATLAS Agents Framework**, providing:

- Standardized agent communication protocol
- Coordinator / orchestration layer
- Message-based architecture
- Pipeline execution
- Modular agent system

---

## Robotics

- ROS2 *(planned)*
- Isaac Sim *(planned)*
- Webots
- OpenCV

---

## Backend

- FastAPI
- WebSockets
- Server-Sent Events (SSE)

---

## Frontend

- React
- TypeScript
- Tailwind CSS

---

## Memory

- SQLite
- Custom Memory Engine
- Short-Term Memory
- Long-Term Memory

---

# Core Components

## Vision Agent

Responsible for understanding the robot's environment.

Capabilities:

- Object detection
- Multi-object tracking
- Bounding box generation
- Object localization

Current implementation:

- YOLO11
- ByteTrack

---

## Audio Agent

Continuously listens for user commands.

Capabilities:

- Speech-to-text
- Streaming transcription
- Voice command parsing

Current implementation:

- Faster-Whisper

---

## Planner Agent

The cognitive reasoning engine of BAI.

Responsibilities:

- Interpret user goals
- Break complex objectives into executable tasks
- Generate execution pipelines
- Coordinate specialized agents

Current implementation:

- Qwen

---

## Memory Agent

Provides persistent memory for the robot.

Features:

- Short-term working memory
- Long-term persistent storage
- Automatic memory promotion
- Memory retrieval
- Memory updates

Backed by the custom Memory Engine.

---

## Execution Agent

Interfaces with the robot or simulator.

Responsibilities:

- Execute movement commands
- Perform robot actions
- Interface with robotics APIs
- Execute planner-generated tasks

---

## World Model

The World Model maintains BAI's understanding of its environment.

It continuously updates information received from the Vision Agent and stores:

- tracked objects
- object positions
- object identities
- visibility state
- spatial relationships

Unlike the Memory Agent, which stores persistent knowledge, the World Model represents the robot's current understanding of the physical world.

---

# Memory Architecture

```text
                 Memory Agent

                       │

                Memory Manager

               ┌───────────────┐
               │               │

      Short-Term Memory   Long-Term Memory

               │               │

           In-Memory        SQLite

               │               │

               └──────┬────────┘

                      ▼

              Planner Agent
```

---

# Agent Communication

All agents communicate through a standardized message protocol provided by the ATLAS Agents Framework.

```text
Planner Agent

      │

 Message

      │

Coordinator

      │

Vision Agent

      │

Response Message

      │

Planner Agent
```

This architecture allows components to remain loosely coupled while enabling new agents to be added with minimal changes to the overall system.

---

# Current Repository Structure

```text
BAI/

├── backend/
│
├── frontend/
│
├── memory_engine/
│
├── atlas_agents/
│
├── simulations/
│
├── docs/
│
└── tests/
```

*(Directory structure subject to change as development continues.)*

---

# Installation

```bash
git clone https://github.com/aparikh27/BAI.git
cd BAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Development Roadmap

## Phase 1 — Core Infrastructure

- [x] Project initialization
- [x] FastAPI backend
- [x] React dashboard
- [x] Live video streaming
- [x] YOLO object detection
- [x] Multi-object tracking

---

## Phase 2 — Autonomous Intelligence

- [x] Speech recognition
- [x] LLM task planner
- [x] Memory Engine
- [x] Multi-Agent Framework
- [ ] Dynamic pipeline generation
- [ ] Planner improvements

---

## Phase 3 — Robotics

- [ ] World model integration
- [ ] Robot execution engine
- [ ] Navigation
- [ ] ROS2 integration
- [ ] Isaac Sim integration
- [ ] Webots integration

---

## Phase 4 — Advanced AI

- [ ] Semantic memory search
- [ ] Vision-language models
- [ ] Self-improving planning
- [ ] Autonomous exploration
- [ ] Multi-robot collaboration

---

# Future Goals

Planned capabilities include:

- Long-term autonomous operation
- Semantic world understanding
- Multi-agent collaboration
- Multi-robot coordination
- Natural language interaction
- Vision-language reasoning
- Reinforcement learning integration
- Distributed AI systems

---

# Related Project

BAI is built using the **ATLAS Agents Framework**, a reusable multi-agent orchestration framework that provides:

- standardized agent interfaces
- message protocol
- coordinator
- pipeline execution
- agent orchestration

ATLAS enables BAI to separate perception, reasoning, memory, and execution into independent, modular agents.

---

# License

MIT License