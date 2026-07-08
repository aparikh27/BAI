from backend.speech.speech_model import SpeechModel
import whisper

class WhisperModel(SpeechModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = whisper.load_model(model_name)

    def process_audio(self, audio_data) -> str:
        """
        Transcribes audio data.
        audio_data can be a string path to a file (e.g., 'sample.wav') 
        or a normalized float32 NumPy array.
        """
        result = self.model.transcribe(audio_data)
        
        return result.get("text", "").strip()