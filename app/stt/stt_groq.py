import os
import io
from groq import Groq
from app.utils.logger import logger

def get_client():
    """Initialize Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set.")
        raise ValueError("GROQ_API_KEY is required for Groq API access.")
    return Groq(api_key=api_key)

def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe audio bytes using Groq Whisper large-v3-turbo model.
    
    Args:
        audio_bytes: Raw audio data in supported format (e.g., WAV, 16kHz mono).
        language: ISO-639-1 language code (default: "en").
    
    Returns:
        The transcribed text.
    """
    try:
        client = get_client()
        file_like = io.BytesIO(audio_bytes)
        
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", file_like),
            model="whisper-large-v3-turbo",
            language=language,
            response_format="text",
            temperature=0.0
        )
        
        logger.info(f"Transcription completed successfully: {transcription[:50]}...")
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

def transcribe_with_timestamps(audio_bytes: bytes, language: str = "en") -> dict:
    """
    Transcribe audio bytes with timestamps using Groq Whisper large-v3-turbo model.
    
    Args:
        audio_bytes: Raw audio data in supported format (e.g., WAV, 16kHz mono).
        language: ISO-639-1 language code (default: "en").
    
    Returns:
        Dictionary containing transcription details including segments with timestamps.
    """
    try:
        client = get_client()
        file_like = io.BytesIO(audio_bytes)
        
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", file_like),
            model="whisper-large-v3-turbo",
            language=language,
            response_format="verbose_json",
            temperature=0.0,
            timestamp_granularities=["segment"]
        )
        
        logger.info("Transcription with timestamps completed successfully.")
        return transcription.to_dict()
    except Exception as e:
        logger.error(f"Error during timestamped transcription: {str(e)}")
        raise
