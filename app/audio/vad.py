import io
import numpy as np
from scipy.io import wavfile
from typing import Optional, Generator
from app.audio.stream import AudioStreamer
from app.utils.logger import logger
from app.stt.stt_groq import transcribe_audio

def chunk_to_wav_bytes(chunk: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Convert numpy int16 audio chunk to WAV bytes.
    
    Args:
        chunk: Numpy array of shape (samples, channels) dtype=int16.
        sample_rate: Audio sample rate.
    
    Returns:
        WAV file bytes.
    """
    bio = io.BytesIO()
    if chunk.ndim == 2 and chunk.shape[1] == 1:
        chunk = chunk.flatten()
    elif chunk.ndim != 1:
        logger.error(f"Invalid chunk shape: {chunk.shape}")
        raise ValueError("Chunk must be mono audio")
    
    # Log audio stats for debugging
    logger.debug(f"Chunk stats: shape={chunk.shape}, dtype={chunk.dtype}, max={chunk.max()}, min={chunk.min()}")
    
    wavfile.write(bio, sample_rate, chunk.astype(np.int16))
    wav_bytes = bio.getvalue()
    logger.debug(f"Generated WAV bytes: length={len(wav_bytes)}")
    return wav_bytes

class VADProcessor:
    """
    Voice Activity Detection using Groq STT API with hysteresis.
    Detects speech segments by transcribing chunks and checking for non-empty text.
    """
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 2048, language: str = "en"):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language = language
        self.buffer = b''
        self.is_speaking = False
        self.speech_count = 0
        self.no_speech_count = 0
        self.speech_start_threshold = 2
        self.speech_end_threshold = 3  # Reduced for faster segment completion
        logger.info("VAD Processor initialized with STT-based detection.")

    def _is_speech_chunk(self, chunk: np.ndarray) -> bool:
        """Check if a chunk contains speech by transcribing it."""
        try:
            wav_bytes = chunk_to_wav_bytes(chunk, self.sample_rate)
            transcript = transcribe_audio(wav_bytes, language=self.language)
            is_speech = len(transcript.strip()) > 0
            logger.debug(f"VAD check: transcript='{transcript.strip()[:20]}...', is_speech={is_speech}")
            return is_speech
        except Exception as e:
            logger.error(f"Error in VAD transcription: {e}")
            return False

    def process_chunk(self, chunk: np.ndarray) -> Optional[bytes]:
        """
        Process a single audio chunk and return speech segment if detected.
        
        Args:
            chunk: Numpy array of audio data.
            
        Returns:
            WAV bytes of speech segment if a complete segment is detected, else None.
        """
        is_speech = self._is_speech_chunk(chunk)

        if not self.is_speaking:
            if is_speech:
                self.speech_count += 1
                if self.speech_count >= self.speech_start_threshold:
                    self.is_speaking = True
                    self.buffer = chunk_to_wav_bytes(chunk, self.sample_rate)
                    self.no_speech_count = 0
                    logger.info("Speech detection started.")
                    return None
                else:
                    self.buffer = b''
            else:
                self.speech_count = 0
                self.buffer = b''
            return None
        else:
            self.buffer += chunk_to_wav_bytes(chunk, self.sample_rate)[44:]
            if is_speech:
                self.no_speech_count = 0
            else:
                self.no_speech_count += 1
                if self.no_speech_count >= self.speech_end_threshold:
                    self.is_speaking = False
                    segment_wav = self._reconstruct_wav(self.buffer)
                    self.buffer = b''
                    logger.info("Speech segment ended.")
                    return segment_wav
            return None

    def _reconstruct_wav(self, raw_pcm_bytes: bytes) -> bytes:
        """Reconstruct full WAV from concatenated raw PCM bytes."""
        bio = io.BytesIO()
        bio.write(b'RIFF')
        bio.write((36 + len(raw_pcm_bytes)).to_bytes(4, 'little'))
        bio.write(b'WAVE')
        bio.write(b'fmt ')
        bio.write((16).to_bytes(4, 'little'))
        bio.write((1).to_bytes(2, 'little'))  # PCM
        bio.write((1).to_bytes(2, 'little'))  # Mono
        bio.write(self.sample_rate.to_bytes(4, 'little'))
        bio.write((self.sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
        bio.write((2).to_bytes(2, 'little'))  # Block align
        bio.write((16).to_bytes(2, 'little'))  # Bits per sample
        bio.write(b'data')
        bio.write(len(raw_pcm_bytes).to_bytes(4, 'little'))
        bio.write(raw_pcm_bytes)
        return bio.getvalue()

    def process_stream(self, streamer: 'AudioStreamer') -> Generator[bytes, None, None]:
        """
        Process the audio stream from streamer and yield speech segments.
        
        Args:
            streamer: AudioStreamer instance.
        """
        while streamer.is_running:
            try:
                chunk = streamer.get_chunk(timeout=0.2)
                segment = self.process_chunk(chunk)
                if segment:
                    yield segment
            except Exception as e:
                logger.error(f"Error processing stream chunk: {e}")
                continue
        if self.buffer and self.is_speaking:
            segment = self._reconstruct_wav(self.buffer[44:])
            yield segment
            logger.info("Final speech segment yielded.")
