# ADR-002: Speech Recognition Module

**Status:** Accepted (Initial Development)

**Date:** July 8, 2026

---

# Background

Speech recognition enables BAI to interact naturally with users through spoken language.

Rather than relying on keyboard input, users should be able to issue voice commands such as:

- "Find the bottle."
- "Count the number of people."
- "Where is my backpack?"
- "Follow me."
- "Stop."

The speech recognition module converts spoken audio into text, which is then interpreted by the task planner.

Speech recognition is intentionally separated from natural language understanding. Its only responsibility is accurate transcription.

---

# Requirements

The selected speech recognition model should satisfy the following requirements:

- Real-time transcription
- High transcription accuracy
- Easy Python integration
- Open-source
- Active community support
- Replaceable implementation
- Future support for embedded deployment
- Offline inference preferred

---

# Candidate Models

## Option 1 — Whisper

### Pros

- Industry-standard speech recognition model
- Developed by OpenAI
- Excellent transcription accuracy
- Supports dozens of languages
- Large developer community
- Extensive documentation
- Multiple model sizes (Tiny, Base, Small, Medium, Large)

### Cons

- Larger models require significant computational resources
- Higher inference latency on CPUs
- May require quantization for embedded deployment

---

## Option 2 — Moonshine

### Pros

- Designed specifically for on-device speech recognition
- Extremely fast inference
- Low memory usage
- Excellent choice for Raspberry Pi and edge devices
- Optimized for low-power deployment

### Cons

- Smaller community
- Less mature ecosystem
- Lower transcription accuracy than Whisper in challenging environments

---

## Option 3 — NVIDIA Canary

### Pros

- State-of-the-art multilingual speech recognition
- Excellent transcription quality
- Supports speech translation
- Strong enterprise support

### Cons

- Larger computational requirements
- More complex deployment
- Primarily optimized for NVIDIA hardware

---

## Option 4 — Qwen Audio

### Pros

- Multimodal language model
- Can reason about spoken language
- Supports conversational interactions
- Potential future integration with an LLM planner

### Cons

- Significantly larger model
- Higher inference latency
- More computationally expensive than dedicated ASR models
- Better suited for higher-level reasoning than pure transcription

---

# Decision

The initial implementation will use **Whisper**.

---

# Rationale

The primary goal of the first development milestone is rapid software development rather than hardware optimization.

Whisper provides:

- Excellent transcription quality
- Mature Python API
- Extensive documentation
- Large developer community
- Proven reliability

Although Moonshine is likely the better deployment choice for embedded hardware, Whisper minimizes development risk and allows the project to focus on higher-level robotics software.

---

# High-Level Architecture

```text
Microphone
     │
     ▼
SpeechRecognizer
     │
     ▼
Transcript
     │
     ▼
Task Planner
     │
     ▼
Robot Action
```

---

# UML Class Diagram

```text
                 +--------------------------+
                 |    SpeechRecognizer      |
                 |--------------------------|
                 | +transcribe(audio)       |
                 +------------▲-------------+
                              |
               implements      |
                              |
                 +------------+-------------+
                 | WhisperRecognizer        |
                 |--------------------------|
                 | -model                   |
                 |--------------------------|
                 | +transcribe(audio)       |
                 +--------------------------+
```

---

# Sequence Diagram

```text
Microphone

    │

    │ audio

    ▼

SpeechRecognizer

    │

    │ transcribe()

    ▼

Transcript

    │

    ▼

Planner

    │

    ▼

Robot
```

---

# Design Decisions

## Abstract Speech Interface

Speech recognition should be accessed through a common interface rather than directly invoking Whisper APIs.

This allows future replacement without affecting the planner.

```text
Planner

↓

SpeechRecognizer

↓

Whisper
```

Later:

```text
Planner

↓

SpeechRecognizer

↓

Moonshine
```

The planner remains unchanged.

---

## Single Responsibility

The speech module is responsible only for converting speech into text.

It is **not responsible** for:

- Command parsing
- Intent recognition
- Task planning
- Robot execution

Those responsibilities belong to downstream modules.

---

## Stateless Design

The speech recognizer does not maintain conversational history.

Each transcription request is processed independently.

Conversation state will eventually be maintained by the planner.

---

# Assumptions

Current assumptions include:

- One microphone input.
- One speaker at a time.
- English-only transcription during initial development.
- Audio is processed in short command segments.
- Offline inference is preferred whenever possible.
- The speech recognizer only outputs text.

---

# Future Considerations

Potential future models include:

- Moonshine
- NVIDIA Canary
- Qwen Audio
- Whisper Turbo
- Faster-Whisper

Benchmarking should compare:

- Word Error Rate (WER)
- Inference latency
- CPU utilization
- GPU utilization
- Memory usage
- Power consumption
- Raspberry Pi performance
- Jetson performance

---

# Risks

- Whisper may not achieve real-time performance on embedded CPUs.
- Background noise may reduce transcription accuracy.
- Long-form transcription introduces additional latency.
- Embedded deployment may require replacing Whisper with a lighter-weight model.

---

# Follow-Up Tasks

- [ ] Implement a SpeechRecognizer interface.
- [ ] Integrate Whisper.
- [ ] Record microphone input.
- [ ] Return transcribed text.
- [ ] Connect the transcript to the task planner.
- [ ] Benchmark alternative speech models.