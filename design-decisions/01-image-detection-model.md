# Design Decision 001: Object Detection Model

**Status:** Accepted (Initial Development)

**Date:** July 1, 2026

---

# Background

The first major component of ATLAS is the perception system, which enables the robot to understand its environment through a live webcam feed.

This perception system will serve as the foundation for all future functionality, including:

- Object tracking
- World modeling
- Speech-guided task execution
- Autonomous task planning
- Robot control
- Robotics simulation

The goal of this decision is to select an object detection model that enables rapid development while remaining flexible enough to support future deployment on embedded hardware such as a Raspberry Pi or NVIDIA Jetson.

---

# Requirements

The selected model should satisfy the following requirements:

- Real-time object detection
- High detection accuracy
- Easy Python integration
- Open-source
- Active community support
- Well-documented API
- Easily replaceable with other detection models
- Potential compatibility with embedded hardware

---

# Candidate Models

## Option 1 — YOLO11

### Pros

- Industry-standard object detection model
- Excellent balance between speed and accuracy
- Very easy to integrate using the Ultralytics Python API
- Large open-source community
- Extensive documentation and tutorials
- Multiple model sizes available (Nano, Small, Medium, Large)

### Cons

- Larger models require more computational resources
- Higher CPU/GPU usage than lightweight alternatives
- May require optimization for Raspberry Pi deployment

---

## Option 2 — MobileNet SSD

### Pros

- Designed specifically for mobile and embedded devices
- Fast CPU inference
- Low memory usage
- Lower power consumption
- Well suited for Raspberry Pi deployment

### Cons

- Lower detection accuracy than YOLO
- Smaller community and ecosystem
- Fewer modern features
- Less flexibility for future expansion

---

## Option 3 — Segment Anything Model (SAM)

### Pros

- State-of-the-art image segmentation
- Can segment objects outside of predefined classes
- Useful for robotics tasks requiring precise object boundaries
- Strong potential for future grasp planning

### Cons

- Performs segmentation rather than object detection
- Computationally intensive
- Difficult to run in real time on embedded hardware
- Better suited as a complementary model rather than the primary perception model

---

# Decision

The initial implementation will use **YOLO11**.

---

# Rationale

Although MobileNet SSD is more suitable for deployment on low-power embedded devices, the primary objective of the first development milestone is rapid software development rather than hardware optimization.

YOLO11 offers:

- Higher detection accuracy
- Better documentation
- A mature Python API
- A larger developer community
- Faster development with fewer integration challenges

This allows development to focus on building the overall robotics software architecture before optimizing for embedded deployment.

---

# Future Considerations

The perception system should be designed so that the object detection model can be replaced without affecting the rest of the software.

Future models to evaluate include:

- YOLO11 Nano
- MobileNet SSD
- Grounding DINO
- OWLv2
- Florence-2

Benchmarking should compare:

- Detection accuracy
- Frames per second (FPS)
- CPU utilization
- GPU utilization
- Memory usage
- Inference latency
- Power consumption
- Raspberry Pi performance
- NVIDIA Jetson performance

---

# Risks

- Larger YOLO11 models may not achieve acceptable performance on Raspberry Pi hardware.
- Additional optimization techniques such as quantization or model pruning may be required.
- Embedded deployment may ultimately require replacing YOLO11 with a lighter-weight detector.

---

# Follow-Up Tasks

- [ ] Integrate YOLO11 into the perception pipeline.
- [ ] Measure inference latency on development hardware.
- [ ] Design a common detector interface to support interchangeable models.
- [ ] Benchmark lightweight models during future development.