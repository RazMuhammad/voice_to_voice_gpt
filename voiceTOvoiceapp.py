import os
import streamlit as st
import whisper
from gtts import gTTS
from groq import Groq
import tempfile
import shutil
YOUR_API_KEY = st.secrets['YOUR_API_KEY']
# Load Whisper model
model = whisper.load_model("base")

# Initialize Groq API client (replace with your actual initialization)
client = Groq(api_key="YOUR_API_KEY")

def chatbot(audio=None, text_input=""):
    if audio is None and not text_input.strip():
        return "No input detected. Please provide either an audio or text input.", None

    if audio:
        # Transcribe the audio input using Whisper
        transcription = model.transcribe(audio)
        user_input = transcription.get("text", "")
    else:
        user_input = text_input.strip()

    if not user_input:
        return "Could not understand input.", None

    # Generate a response using Llama 8B via Groq API
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}", None

    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        audio_path = tmp_file.name

    return response_text, audio_path

# Streamlit app
def main():
    st.title("Voice-to-Voice & Text Chatbot")
    st.subheader("Powered by OpenAI Whisper, Llama 8B, and gTTS")
    st.write("Talk to the AI-powered chatbot using your voice or text, and get responses in real-time.")

    # File uploader for audio input
    audio_input = st.file_uploader("Record Your Voice", type=["wav", "mp3"])
    text_input = st.text_input("Or Type Your Message Here")

    if st.button("Submit"):
        if audio_input is not None:
            audio_path = audio_input
        else:
            audio_path = None

        response_text, audio_path = chatbot(audio=audio_path, text_input=text_input)

        st.write("Chatbot Response:")
        st.write(response_text)

        if audio_path:
            st.audio(audio_path)

# Launch the Streamlit app
if __name__ == "__main__":
    main()
