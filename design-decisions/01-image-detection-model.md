# ADR-001: Perception System Architecture

**Status:** Accepted

**Date:** July 1, 2026

---

# Background

BAI is an AI-powered robotics platform capable of perceiving its environment, maintaining an internal world model, understanding voice commands, and autonomously planning actions.

The perception system is the foundation of the project. Every downstream subsystem—including world modeling, speech-guided task execution, planning, and robotics simulation—depends on accurate and reliable perception.

The primary objective of this design is to create a modular architecture where individual components can evolve independently without requiring changes throughout the codebase.

---

# Functional Requirements

The perception subsystem must support:

- Real-time webcam input
- Object detection
- Object tracking across frames
- Persistent object identities
- World model updates
- Future dashboard streaming
- Future speech-guided interaction
- Future ROS2 integration

---

# Non-Functional Requirements

- Modular architecture
- Replaceable detection models
- Loose coupling between subsystems
- Real-time performance
- Python-first implementation
- Open-source dependencies
- Support for future embedded deployment

---

# High-Level Architecture

```text
                Webcam
                   │
                   ▼
           YOLODetector
                   │
                   ▼
         FrameDetections
                   │
                   ▼
            World Model
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Dashboard        Planner (Future)
                            │
                            ▼
                     Robot Actions
```

---

# Component Responsibilities

## YOLODetector

Responsibilities:

- Capture video frames
- Execute YOLO inference
- Perform ByteTrack object tracking
- Produce typed Detection objects
- Return one FrameDetections object per frame

The detector **does not maintain long-term state**.

Its responsibility ends after processing the current frame.

---

## Detection

Each detected object contains:

- Class ID
- Class name
- Confidence
- Bounding box
- Frame index
- Tracking ID

```text
Detection
──────────────
class_id
class_name
confidence
box
frame_index
track_id
```

---

## FrameDetections

Each processed frame is represented by a collection of detections.

```text
FrameDetections
──────────────────────
frame_index
detections[]
```

This allows downstream systems to reason about an entire scene rather than individual detections.

---

## World Model

The World Model maintains persistent knowledge about the environment.

Responsibilities:

- Store tracked objects
- Update object positions
- Remember previously observed objects
- Maintain object visibility
- Serve as the source of truth for planners

Unlike the detector, the World Model persists across frames.

---

# Object Lifetime

```text
Frame 1

Bottle (ID 4)

↓

World
ID 4
visible = true

↓

Frame 2

Bottle (ID 4)

↓

Update

↓

Frame 10

Bottle disappears

↓

World

ID 4
visible = false
last_seen_frame = 10
```

This allows the robot to remember objects after they leave the camera view.

---

# UML Class Diagram

```text
                    +----------------------+
                    |   ImageDetector      |
                    |----------------------|
                    | +process_frame()     |
                    +----------▲-----------+
                               |
                               |
                    +----------+-----------+
                    |     YOLODetector     |
                    |----------------------|
                    | -model : YOLO        |
                    |----------------------|
                    | +process_frame()     |
                    | +detect()            |
                    +----------+-----------+
                               |
                creates        |
                               ▼
                  +----------------------+
                  |     Detection        |
                  |----------------------|
                  | class_id             |
                  | class_name           |
                  | confidence           |
                  | box                  |
                  | frame_index          |
                  | track_id             |
                  +----------------------+

                               ▲

                               |

                  +----------------------+
                  |  FrameDetection      |
                  |----------------------|
                  | frame_index          |
                  | detections[]         |
                  +----------------------+

                               |

                               ▼

                  +----------------------+
                  |    World             |
                  |----------------------|
                  | memory               |
                  |----------------------|
                  | update()             |
                  | get_visible_objects()|
                  | get_object_by_track  |
                  | _id(track_id: int)   |
                  +----------------------+
```

---

# Sequence Diagram

```text
Camera

 │

 │ frame

 ▼

YOLODetector

 │

 │ process_frame()

 ▼

FrameDetections

 │

 │ update()

 ▼

WorldModel

 │

 │ current world state

 ▼

Planner (future)

 │

 ▼

Robot
```

---

# Design Decisions

## Typed Data Models

Instead of exposing raw Ultralytics objects throughout the application, the detector converts every prediction into custom dataclasses.

Advantages:

- Model-independent interface
- Easier testing
- Cleaner APIs
- Strong typing
- Simpler serialization

---

## Frame-Based Processing

The detector emits one FrameDetections object per frame rather than individual detections.

Advantages:

- Represents the complete scene
- Simplifies world model updates
- Supports future planning algorithms
- Enables frame-level statistics

---

## Persistent World Model

The detector remains stateless.

Persistent information is maintained exclusively inside the World Model.

Advantages:

- Separation of concerns
- Easier debugging
- Future support for planning
- Supports memory and reasoning

---

## Model Abstraction

The project defines an abstract ImageDetector interface.

Concrete implementations include:

- YOLO11
- MobileNet SSD (future)
- GroundingDINO (future)
- Florence-2 (future)

Replacing the detector should not require changes elsewhere in the system.

---

# Assumptions

Current assumptions include:

- One primary camera input.
- ByteTrack provides stable tracking IDs.
- Each tracked object has a unique ID.
- The world model is the single source of truth for object state.
- Object identity persists until explicitly removed.
- One frame is processed at a time.
- Real-time performance is prioritized over maximum detection accuracy during early development.

---

# Risks

- Tracking IDs may change after prolonged occlusion.
- Embedded deployment may require replacing YOLO11 with a lighter detector.
- Long-running sessions may require world model pruning.
- Future multi-camera support will require redesigning object identity management.

---

# Future Work

- Dashboard video streaming
- React visualization
- Speech recognition
- LLM task planner
- ROS2 integration
- Robotics simulation
- Multi-camera perception
- Sensor fusion
- Semantic mapping