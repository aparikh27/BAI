from backend.speech.speech_model import SpeechModel
import numpy as np
import whisper


class WhisperModel(SpeechModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_name)
        return self.model

    def process_audio(self, audio_data) -> str:
        """
        Transcribes audio data.
        audio_data can be a string path to a file (e.g., 'sample.wav')
        or a normalized float32 NumPy array.
        """
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.astype(np.float32, copy=False)
            if audio_data.ndim > 1:
                audio_data = audio_data.reshape(-1)

        result = self._get_model().transcribe(audio_data)

        return result.get("text", "").strip()
