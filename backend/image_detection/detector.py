from abc import ABC, abstractmethod


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