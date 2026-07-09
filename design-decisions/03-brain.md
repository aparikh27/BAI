# ADR-003: Agent Planning System

**Status:** Accepted (Initial Development)

**Date:** July 9, 2026

---

# Background

BAI is an AI-powered robotics platform capable of perceiving its environment, maintaining an internal world model, understanding spoken commands, and autonomously determining how to accomplish user requests.

The perception system identifies and tracks objects, while the speech recognition system converts spoken language into text. However, neither component understands **what the robot should do**.

The purpose of the Agent Planning System is to bridge this gap by interpreting user commands, reasoning over the current world model, and producing a structured action plan that downstream systems can execute.

The planner represents the cognitive layer of the robot.

---

# Requirements

The planning system should satisfy the following requirements:

- Understand natural language commands
- Determine user intent
- Reason using the current world model
- Generate structured action plans
- Support future autonomous behaviors
- Remain independent of any specific LLM implementation
- Support local inference
- Minimize coupling with perception and robot execution

---

# Candidate Models

## Option 1 — GLM-5.2

### Pros

- Excellent reasoning performance
- Designed for agentic workflows
- Strong structured output capabilities
- Open-source
- Local deployment
- Well suited for robotics planning
- Supports tool-based reasoning

### Cons

- Larger computational requirements than lightweight models
- May require quantization for embedded deployment

---

## Option 2 — Kimi K2.5

### Pros

- Excellent reasoning ability
- Strong coding capabilities
- Long context window
- Open-source
- Local deployment

### Cons

- Optimized primarily for coding and general reasoning
- Less focused on agentic planning than GLM

---

## Option 3 — Llama 3.3 70B

### Pros

- Large open-source ecosystem
- Strong instruction following
- Mature tooling
- Proven reasoning performance

### Cons

- Very large memory requirements
- Difficult to run locally without high-end hardware
- Higher inference latency

---

## Option 4 — DeepSeek-R

### Pros

- Excellent reasoning performance
- Strong planning capabilities
- Open-source

### Cons

- Computationally intensive
- Slower inference than smaller alternatives

---

# Decision

The initial implementation will use **GLM-5.2**.

The planner will be abstracted behind a common interface so that future implementations can replace GLM with another language model without affecting the remainder of the system.

---

# Rationale

The primary objective of the first development milestone is to create an autonomous robotics platform capable of running entirely with open-source software.

GLM-5.2 provides:

- Excellent reasoning ability
- Strong support for agentic workflows
- High-quality structured outputs
- Local deployment
- No recurring API costs
- No internet dependency
- Future compatibility with embedded robotics systems

Although cloud-hosted language models may achieve higher overall reasoning performance, local deployment better aligns with BAI's long-term goals of privacy, reliability, cost efficiency, and autonomous execution.

---

# High-Level Architecture

```text
                 Camera
                    │
                    ▼
           Perception System
                    │
                    ▼
              World Model
                    ▲
                    │
Microphone          │
     │              │
     ▼              │
Speech Recognition  │
     │              │
     └──────────────┐
                    ▼
             Agent Planner
                    │
                    ▼
              Action Plan
                    │
                    ▼
        Robot Controller (Future)
```

---

# Responsibilities

The Agent Planner is responsible for:

- Understanding user intent
- Determining whether a request is actionable
- Selecting relevant objects from the world model
- Reasoning about how to accomplish the requested task
- Producing a structured action plan

The planner is **not responsible** for:

- Speech recognition
- Object detection
- Object tracking
- World model updates
- Robot motion
- Motor control
- Camera processing

---

# Planning Pipeline

```text
Speech Transcript

        +

Current World Model

        │

        ▼

Prompt Construction

        │

        ▼

GLM-5.2

        │

        ▼

Structured Action Plan
```

---

# UML Class Diagram

```text
                    +----------------------+
                    |    AgentPlanner      |
                    |----------------------|
                    | +plan()              |
                    +----------▲-----------+
                               |
                 implements    |
                               |
                +--------------+---------------+
                |      GLMPlanner              |
                |------------------------------|
                | -model                       |
                |------------------------------|
                | +plan()                      |
                +--------------+---------------+
                               |
                               |
                               ▼
                    +----------------------+
                    |     ActionPlan       |
                    |----------------------|
                    | intent               |
                    | target               |
                    | steps[]              |
                    +----------------------+
```

---

# Sequence Diagram

```text
User

 │

 │ "Find the bottle"

 ▼

Speech Recognition

 │

 ▼

Transcript

 │

 │ + Current World Model

 ▼

Agent Planner

 │

 ▼

Action Plan

 │

 ▼

Robot Controller (Future)
```

---

# Design Decisions

## Planner Abstraction

The planner is accessed through an abstract interface.

```python
class Planner(ABC):

    @abstractmethod
    def plan(self, command, world_model):
        pass
```

Concrete implementations may include:

- GLMPlanner
- KimiPlanner
- LlamaPlanner
- DeepSeekPlanner

This architecture prevents the remainder of the system from depending on any specific language model.

---

## Structured Outputs

The planner returns structured data rather than natural language.

Example:

```json
{
    "intent": "find_object",
    "target": "bottle",
    "steps": [
        "Locate bottle",
        "Rotate toward bottle",
        "Move forward"
    ]
}
```

Structured outputs eliminate additional parsing and simplify downstream execution.

---

## World Model Context

The planner does not receive raw camera frames or microphone audio.

Instead, it receives a summarized representation of the current world.

Example:

```text
Visible Objects

- person (ID 1)
- bottle (ID 2)
- chair (ID 3)

User Command

Find the bottle.
```

Providing structured context significantly reduces token usage while improving planning quality.

---

## Separation of Concerns

The planner determines **what** the robot should do.

It does not determine **how** motors should move or trajectories should be executed.

Low-level execution belongs to the Robot Controller.

---

## Stateless Planning

Each planning request is independent.

The planner does not maintain conversational history or persistent memory.

Persistent knowledge about the environment remains within the World Model.

---

# Assumptions

Current assumptions include:

- One user issues commands at a time.
- English is the only supported language.
- The World Model accurately reflects the current environment.
- Only currently visible tracked objects are provided to the planner.
- One action plan is generated for each command.
- Robot execution is handled by downstream systems.

---

# Risks

- Incorrect plans resulting from incomplete world models.
- Hallucinated action plans.
- Local inference latency on embedded hardware.
- Large models may require quantization for deployment.
- Future autonomous operation may require additional safety constraints before executing generated plans.

---

# Future Considerations

Potential future planner implementations include:

- Kimi Planner
- Llama Planner
- DeepSeek Planner
- Multi-agent planning
- Hierarchical task planning
- Dynamic replanning
- Long-term memory integration
- Tool calling
- ROS2 integration
- Failure recovery
- Multi-step autonomous task execution

---

# Follow-Up Tasks

- [ ] Create an abstract `Planner` interface.
- [ ] Implement `GLMPlanner`.
- [ ] Design an `ActionPlan` dataclass.
- [ ] Construct prompts from the World Model.
- [ ] Parse structured JSON responses.
- [ ] Connect the planner to the Robot Controller.
- [ ] Benchmark alternative open-source LLMs.