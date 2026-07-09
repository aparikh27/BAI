from abc import ABC, abstractmethod


class SpeechModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def process_audio(self, audio_data) -> str:
        """Process audio data and return a clean text transcription string."""
        pass