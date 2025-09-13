import queue
import threading
import numpy as np
import sounddevice as sd
from app.utils.logger import logger

class AudioStreamer:
    """
    Real-time audio streamer from microphone.
    Captures audio in chunks of 2048 samples (128ms at 16kHz).
    """
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 2048, channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_running = False
        self.thread = None

    def _callback(self, indata, frames, time, status):
        """Callback function for sounddevice stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        # Put numpy array (int16) into queue
        self.audio_queue.put(indata.copy())

    def start(self):
        """Start the audio stream."""
        if self.is_running:
            logger.warning("Audio stream already running.")
            return
        try:
            self.stream = sd.InputStream(
                callback=self._callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=self.chunk_size
            )
            self.stream.start()
            self.is_running = True
            logger.info("Audio streamer started.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop(self):
        """Stop the audio stream."""
        if not self.is_running:
            return
        try:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            logger.info("Audio streamer stopped.")
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Timeout in seconds for queue.get().
            
        Returns:
            Numpy array of shape (chunk_size, channels) dtype=int16.
        """
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            return chunk
        except queue.Empty:
            return np.zeros((self.chunk_size, self.channels), dtype=np.int16)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
