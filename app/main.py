import io
import sys
from dotenv import load_dotenv
import simpleaudio as sa
from scipy.io import wavfile
from app.audio.stream import AudioStreamer
from app.audio.vad import VADProcessor
from app.stt.stt_groq import transcribe_audio
from app.bot.bot_groq import generate_response
from app.tts.tts_groq import synthesize_speech
from app.utils.logger import logger

def main():
    load_dotenv()
    
    sample_rate = 16000
    chunk_size = 2048
    language = "en"
    tts_voice = "Fritz-PlayAI"
    context_window = 5
    
    streamer = AudioStreamer(sample_rate=sample_rate, chunk_size=chunk_size)
    vad = VADProcessor(sample_rate=sample_rate, chunk_size=chunk_size, language=language)
    
    conversation_history = []
    
    logger.info("Starting Voice AI system. Speak into the microphone. Press Ctrl+C to exit.")
    
    with streamer:
        try:
            for segment in vad.process_stream(streamer):
                try:
                    transcript = transcribe_audio(segment, language=language).strip()
                    if not transcript:
                        continue
                    
                    logger.info(f"User said: {transcript}")
                    
                    response = generate_response(
                        user_message=transcript,
                        conversation_history=conversation_history,
                        context_window=context_window
                    )
                    
                    conversation_history.append({"role": "user", "content": transcript})
                    conversation_history.append({"role": "assistant", "content": response})
                    
                    logger.info(f"Assistant response: {response}")
                    
                    audio_bytes = synthesize_speech(text=response, voice=tts_voice)
                    
                    # Parse WAV bytes
                    bio = io.BytesIO(audio_bytes)
                    rate, data = wavfile.read(bio)
                    
                    # Determine channels
                    channels = data.shape[1] if data.ndim > 1 else 1
                    
                    # Play audio
                    play_obj = sa.play_buffer(
                        data,
                        channels,
                        data.dtype.itemsize,
                        rate
                    )
                    play_obj.wait_done()
                    
                except Exception as e:
                    logger.error(f"Error in processing loop: {str(e)}")
                    continue
                
        except KeyboardInterrupt:
            logger.info("Voice AI system shutting down.")
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
