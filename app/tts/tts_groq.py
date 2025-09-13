import os
from groq import Groq
from app.utils.logger import logger

def get_client():
    """Initialize Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set.")
        raise ValueError("GROQ_API_KEY is required for Groq API access.")
    return Groq(api_key=api_key)

def synthesize_speech(text: str, voice: str = "Fritz-PlayAI", model: str = "playai-tts") -> bytes:
    """
    Synthesize speech from text using Groq PlayAI TTS model.
    
    Args:
        text: The text to convert to speech. Maximum 10K characters.
        voice: The voice to use (e.g., 'Fritz-PlayAI'). Defaults to 'Fritz-PlayAI'.
        model: The TTS model to use (e.g., 'playai-tts' for English). Defaults to 'playai-tts'.
    
    Returns:
        Audio bytes in WAV format.
    """
    if len(text) > 10000:
        raise ValueError("Text length exceeds 10K characters limit.")
    
    try:
        client = get_client()
        
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        logger.info(f"TTS synthesis completed successfully for text: '{text[:50]}...'")
        return response.content
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {str(e)}")
        raise
