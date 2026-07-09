import sounddevice as sd
import numpy as np
import queue
from backend.speech.speech_whisper import WhisperModel

speech_model = WhisperModel(model_name="base")

class ContinuousAudioStream:
    def __init__(self, chunk_duration=3, fs=16000):
        self.chunk_duration = chunk_duration  # Process audio in 3-second blocks
        self.fs = fs
        self.chunk_samples = int(chunk_duration * fs)
        self.audio_queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """This runs internally in a background thread for every mic frame."""
        self.audio_queue.put(indata.copy())

    def stream_and_transcribe(self):
        """Generator that yields transcripts continuously."""
        # Open a live, non-blocking hardware input stream
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', callback=self._callback):
            print("Continuous listening started...")
            
            buffer = []
            samples_collected = 0
            
            while True:
                # Pull raw data chunk from the queue
                data_chunk = self.audio_queue.get()
                buffer.append(data_chunk)
                samples_collected += len(data_chunk)
                
                # Once we have accumulated enough samples for a 3-second block
                if samples_collected >= self.chunk_samples:
                    # Combine chunks and flatten to 1D
                    full_block = np.concatenate(buffer, axis=0).flatten()
                    
                    # Transcribe the rolling chunk
                    transcript = speech_model.process_audio(full_block)
                    
                    if transcript:  # Only yield if the user actually said something
                        yield transcript
                    
                    # Reset buffer for the next window
                    buffer = []
                    samples_collected = 0