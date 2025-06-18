# Voice_Bot (EchoMind – Your Personal AI Voice Assistant )

**EchoMind** is a personalized voice-based assistant powered by Groq’s LLaMA 3.1 LLM, Faster-Whisper STT, and Groq TTS (PlayAI voices). It listens to your voice, understands your intent using your preloaded resume, and responds in a natural, human-like tone.

Built with ❤️ using Streamlit.

---

## 🎯 Features

- 🎙️ Voice input using in-browser microphone
- 🧠 Fast transcription with `faster-whisper`
- 💬 Smart, first-person responses powered by Groq’s LLaMA 3.1 (8B)
- 🔊 Text-to-speech via Groq PlayAI voices
- 📄 Preloaded resume (stored securely in backend)
- 🧑 Behavioral and technical Q&A support
- 🌙 Beautiful dark-themed UI with personalized sidebar

---

## 📁 Project Structure
.
├── app.py # Main Streamlit app

├── assets/

│ └── data.txt 

├── requirements.txt

└── README


---

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/DivyangP2003/Voice_Bot.git

cd Voice_Bot

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Set API Keys
Create a file at .streamlit/secrets.toml with:

GROQ_KEY = "your_groq_api_key"

HUME_API_KEY = "optional_but_unused"

### 4. Add Resume
Save your pre-cleaned resume as plain text to:

assets/data.txt

### 5. Run the App
streamlit run app.py



