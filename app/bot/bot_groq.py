import os
from typing import List, Dict
from groq import Groq
from app.utils.logger import logger

def get_client():
    """Initialize Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set.")
        raise ValueError("GROQ_API_KEY is required for Groq API access.")
    return Groq(api_key=api_key)

def generate_response(user_message: str, conversation_history: List[Dict[str, str]] = None, context_window: int = 5) -> str:
    """
    Generate a conversational response using Groq OpenAI GPT-OSS 120B model.
    
    Args:
        user_message: The user's input message.
        conversation_history: List of previous messages as [{"role": "user/system/assistant", "content": str}].
        context_window: Maximum number of previous turns to include (default: 5).
    
    Returns:
        The model's generated response.
    """
    try:
        client = get_client()
        
        messages = [{"role": "system", "content": "You are a helpful AI assistant for voice conversations."}]
        
        if conversation_history:
            limited_history = conversation_history[-context_window * 2:]
            messages.extend(limited_history)
        
        messages.append({"role": "user", "content": user_message})
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1.0
        )
        
        response = completion.choices[0].message.content.strip()
        logger.info(f"Generated response: {response[:50]}...")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
