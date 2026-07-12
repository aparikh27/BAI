import sounddevice as sd
import numpy as np
import queue
import threading
from backend.speech.speech_whisper import WhisperModel

speech_model = WhisperModel(model_name="tiny")

class ContinuousAudioStream:
    def __init__(self, chunk_duration=3, fs=16000):
        self.chunk_duration = chunk_duration  # Process audio in 3-second blocks
        self.fs = fs
        self.chunk_samples = int(chunk_duration * fs)
        self.audio_queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """This runs internally in a background thread for every mic frame."""
        self.audio_queue.put(indata.copy())

    def stream_and_transcribe(self, stop_event: threading.Event | None = None):
        """Generator that yields transcripts continuously."""
        stop_event = stop_event or threading.Event()

        with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', callback=self._callback):
            print("Continuous listening started...")

            buffer = []
            samples_collected = 0

            while not stop_event.is_set():
                try:
                    # A timeout lets a disconnected SSE client close the microphone
                    # stream even while no audio is arriving.
                    data_chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                buffer.append(data_chunk)
                samples_collected += len(data_chunk)

                if samples_collected >= self.chunk_samples:
                    full_block = np.concatenate(buffer, axis=0).flatten()
                    transcript = speech_model.process_audio(full_block)

                    if transcript:
                        yield transcript

                    buffer = []
                    samples_collected = 0
