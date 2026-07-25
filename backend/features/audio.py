"""
backend/features/audio.py
──────────────────────────
Microphone capture utilities for the BAI project.

``ContinuousAudioStream`` captures live audio from the system mic in
configurable chunks and yields raw numpy buffers.  Transcription is no
longer performed here — the Coordinator pipeline handles it via
``WhisperAudioAgent``.
"""

import queue
import threading

import numpy as np
import sounddevice as sd


class Audio:
    """Simple start/stop flag for legacy audio recording control."""

    def __init__(self, duration: int = 3):
        self.duration = duration
        self.listening = False

    def start_listening(self):
        self.listening = True

    def stop_listening(self):
        self.listening = False


class ContinuousAudioStream:
    """Captures microphone audio in fixed-duration chunks.

    Yields raw ``numpy.ndarray`` buffers (float32, 16 kHz, mono) that can
    be forwarded to a ``WhisperAudioAgent`` through the Coordinator's
    message bus.
    """

    def __init__(self, chunk_duration: int = 3, fs: int = 16000):
        self.chunk_duration = chunk_duration
        self.fs = fs
        self.chunk_samples = int(chunk_duration * fs)
        self.audio_queue: queue.Queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """Sounddevice callback — runs in a background thread for every mic frame."""
        self.audio_queue.put(indata.copy())

    def stream_audio(self, stop_event: threading.Event | None = None):
        """Generator that yields raw audio numpy arrays (float32, mono).

        Each yielded array corresponds to ``chunk_duration`` seconds of audio.
        The caller is responsible for routing these through the Coordinator for
        transcription.
        """
        stop_event = stop_event or threading.Event()

        with sd.InputStream(
            samplerate=self.fs, channels=1, dtype="float32", callback=self._callback
        ):
            print("🎙️ Continuous microphone capture started...")

            buffer: list[np.ndarray] = []
            samples_collected = 0

            while not stop_event.is_set():
                try:
                    data_chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                buffer.append(data_chunk)
                samples_collected += len(data_chunk)

                if samples_collected >= self.chunk_samples:
                    full_block = np.concatenate(buffer, axis=0).flatten()
                    yield full_block

                    buffer = []
                    samples_collected = 0
