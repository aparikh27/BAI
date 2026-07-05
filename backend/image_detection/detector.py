from abc import ABC, abstractmethod

class ImageDetector(ABC):

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def detect(self, source, confidence):
        """Yield detection objects from a video source or frame stream."""
        pass