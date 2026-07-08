from abc import ABC, abstractmethod
from dataclasses import dataclass


class ImageDetector(ABC):

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def process_frame(self, frame, confidence, frame_index):
        """Process a single frame and return detections plus an annotated frame."""
        pass

    @abstractmethod
    def detect(self, source, confidence, frame_callback=None):
        """Yield detection objects from a video source or frame stream."""
        pass

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    frame_index: int
    track_id: int | None = None

    def verbose(self) -> str:
        return (
            f"frame={self.frame_index} class={self.class_name} "
            f"confidence={self.confidence:.2f} box={self.box} track_id={self.track_id}"
        )

@dataclass
class FrameDetection:
    frame_index: int
    detections: list[Detection]