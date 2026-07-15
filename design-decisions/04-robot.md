# ADR-004: Robot Execution Engine

**Status:** Accepted (Initial Development)

**Date:** July 11, 2026

---

# Background

BAI's perception and cognitive pipeline is now capable of:

- Detecting and tracking objects
- Maintaining a world model
- Understanding spoken commands
- Generating high-level action plans

However, these action plans are currently not executable. The system requires an execution layer responsible for translating planner outputs into robot actions.

Rather than directly controlling hardware, the initial implementation will execute commands within a robotics simulation environment.

This allows the complete robotics pipeline to be developed, tested, and validated before deploying to physical hardware.

---

# Requirements

The execution platform should satisfy the following requirements:

- Support virtual robots
- Simulate sensors and actuators
- Execute navigation commands
- Allow future integration with ROS2
- Support Python development
- Be actively maintained
- Be free and open-source
- Enable eventual migration to real hardware

---

# Candidate Platforms

## Option 1 — Webots

### Pros

- Open-source
- Excellent Python API
- Native support for many mobile robots
- Built-in cameras, LiDAR, GPS, IMU, motors, and robot arms
- Integrated physics engine
- Supports ROS2 integration
- Easy to install and configure
- Widely used in robotics education and research

### Cons

- Smaller ecosystem than Gazebo
- Some advanced robotics features are less mature

---

## Option 2 — Gazebo

### Pros

- Industry-standard robotics simulator
- Excellent physics simulation
- Large ROS2 ecosystem
- Highly realistic environments
- Extensive plugin support

### Cons

- Steeper learning curve
- More difficult setup
- Strong dependency on ROS2
- Slower development for first-time robotics projects

---

## Option 3 — Custom Pygame Simulator

### Pros

- Complete control over implementation
- Extremely lightweight
- Simple debugging
- Fast development

### Cons

- No realistic physics
- No sensor simulation
- No robot models
- Difficult transition to real robotics hardware

---

# Decision

The initial implementation will use **Webots**.

The execution engine will communicate through an abstract robot interface, allowing the simulator to be replaced with physical hardware in the future without modifying higher-level software.

---

# Rationale

The primary objective is to demonstrate an end-to-end autonomous robotics pipeline rather than build a hardware-specific application.

Webots provides:

- High-quality robotics simulation
- Native Python support
- Built-in sensors and actuators
- Easy integration with ROS2
- Support for multiple robot platforms
- Smooth migration toward real robotic hardware

Compared to Gazebo, Webots offers a significantly easier learning curve while still providing professional-grade robotics capabilities.

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
Speech ──► Whisper
                  │
                  ▼
           Agent Planner
                  │
                  ▼
            Action Plan
                  │
                  ▼
         Execution Engine
                  │
                  ▼
           Robot Interface
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Webots Robot         Future Hardware
```

---

# Responsibilities

The Execution Engine is responsible for:

- Receiving action plans
- Executing actions sequentially
- Monitoring execution state
- Reporting completion or failure
- Sending movement commands to the robot

The Execution Engine is **not responsible** for:

- Speech recognition
- Object detection
- Object tracking
- World model updates
- Path planning
- High-level reasoning
- Natural language understanding

---

# Execution Pipeline

```text
Action Plan

        │

        ▼

Execution Engine

        │

        ▼

Robot Interface

        │

        ▼

Webots Robot

        │

        ▼

Robot Motion
```

---

# UML Class Diagram

```text
                   +----------------------+
                   |   RobotInterface     |
                   |----------------------|
                   | +move_forward()      |
                   | +rotate()            |
                   | +stop()              |
                   | +pickup()            |
                   +----------▲-----------+
                              |
                  implements  |
                              |
            +-----------------+-----------------+
            |                                   |
            ▼                                   ▼
     WebotsRobot                      PhysicalRobot
```

---

```text
                +----------------------+
                | ExecutionEngine      |
                |----------------------|
                | -robot               |
                |----------------------|
                | +execute(plan)       |
                | +execute_step()      |
                +----------+-----------+
                           |
                           ▼
                     RobotInterface
```

---

# Sequence Diagram

```text
Planner

    │

    │ ActionPlan

    ▼

Execution Engine

    │

    │ move_forward()

    ▼

Robot Interface

    │

    ▼

Webots Robot

    │

    ▼

Robot Moves
```

---

# Design Decisions

## Robot Abstraction

Robot control will be abstracted behind a common interface.

```python
class RobotInterface(ABC):

    @abstractmethod
    def move_forward(self):
        pass

    @abstractmethod
    def rotate(self, angle):
        pass

    @abstractmethod
    def stop(self):
        pass
```

Concrete implementations may include:

- WebotsRobot
- ROSRobot
- RaspberryPiRobot
- PhysicalRobot

This abstraction prevents higher-level software from depending on a specific simulator or hardware platform.

---

## Action-Based Execution

The planner produces high-level actions rather than motor commands.

Example:

```json
{
    "intent": "find_object",
    "target": "bottle",
    "steps": [
        "Rotate",
        "MoveForward",
        "Stop"
    ]
}
```

The Execution Engine interprets these actions and invokes the appropriate robot interface methods.

---

## Sequential Execution

Actions are executed one step at a time.

Each action must complete before the next action begins.

Future versions may support:

- Parallel actions
- Interruptible execution
- Dynamic replanning

---

## Separation of Concerns

The planner determines **what** the robot should do.

The Execution Engine determines **when** actions occur.

The Robot Interface determines **how** robot hardware performs those actions.

---

# Assumptions

Current assumptions include:

- One robot is active.
- One action plan executes at a time.
- Actions execute sequentially.
- The simulator accurately reflects robot movement.
- Path planning is handled manually or by future modules.
- Collision avoidance is outside the scope of the initial implementation.

---

# Risks

- Simulator behavior may differ from physical hardware.
- Robot execution latency may affect responsiveness.
- Future hardware may require additional actuator abstractions.
- More complex behaviors will require replanning and recovery mechanisms.

---

# Future Considerations

Future improvements may include:

- ROS2 integration
- Real robotic hardware
- Path planning
- Obstacle avoidance
- Autonomous navigation
- Multi-robot coordination
- Dynamic replanning
- Grasp planning
- Manipulator arm support
- Sensor fusion

---

# Follow-Up Tasks

- [ ] Create `RobotInterface`.
- [ ] Implement `WebotsRobot`.
- [ ] Create `ExecutionEngine`.
- [ ] Parse planner action sequences.
- [ ] Execute robot actions in Webots.
- [ ] Add execution state tracking.
- [ ] Integrate with ROS2.
- [ ] Support physical robot implementations.