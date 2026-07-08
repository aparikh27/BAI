from abc import ABC, abstractmethod

class SpeechModel(ABC):
    def __init__self__(self, model: str):
        self.model = model
    @abstractmethod
    def process_audio(self, audio_data):
        """Process audio data and return transcriptions or detections."""
        pass