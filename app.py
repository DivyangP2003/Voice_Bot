import streamlit as st
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from groq import Groq
import re
import time
from streamlit.components.v1 import html

# ---- PAGE CONFIG ----
st.set_page_config(page_title="🎙 Voice Bot", layout="centered")

# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background-color: #fff8f0;
        }

        .main {
            background-color: #fffaf3;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 0 12px rgba(0,0,0,0.05);
        }

        h1, h2, h3, h4 {
            color: #ff5e57;
            font-weight: bold;
        }

        .stButton>button {
            background-color: #ff5e57;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 0.5em 1.5em;
            font-weight: bold;
            transition: background-color 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #ff7e75;
        }

        .stSelectbox label {
            font-weight: bold;
            color: #444;
        }

        .transcript-box {
            background-color: #e5f5ff;
            border-left: 5px solid #00bfff;
            padding: 10px;
            border-radius: 10px;
            font-size: 1.1rem;
        }

        .audio-player {
            margin-top: 1rem;
            border: 2px dashed #ffbd59;
            padding: 1rem;
            background-color: #fff3e0;
            border-radius: 12px;
        }

        audio {
            width: 100%;
            outline: none;
        }

        @keyframes float {
            0% { transform: translatey(0px); }
            50% { transform: translatey(-10px); }
            100% { transform: translatey(0px); }
        }
    </style>
""", unsafe_allow_html=True)

# ---- CONFIG ----
HUME_API_KEY = st.secrets["HUME_API_KEY"]  # Unused
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

# ---- WHISPER MODEL LOADING ----
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper_model()

# ---- GROQ CLIENT ----
@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

groq_client = load_groq_client()

# ---- PAGE TITLE ----
st.title("🎙 Personalized Voice Bot (Groq + TTS + Whisper)")

# ---- FUNCTIONS ----
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()

def transcribe_audio_faster_whisper(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        segments, _ = whisper_model.transcribe(tmpfile.name)
        return " ".join(segment.text for segment in segments)

def generate_response_groq_direct(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.5
    }
    try:
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception:
        st.error("⚠ Error generating response from Groq.")
        st.stop()

def parse_groq_wait_time(wait_str):
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?'
    match = re.match(pattern, wait_str)
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = float(match.group(3) or 0.0)
    return h * 3600 + m * 60 + s

def synthesize_tts_file(text, voice="Fritz-PlayAI", fmt="wav"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "playai-tts",
        "input": text,
        "voice": voice,
        "response_format": fmt
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/audio/speech", headers=headers, json=payload)
        if resp.status_code == 429:
            msg = resp.json()['error'].get('message', '')
            wait_str = re.search(r'in ([\dhms\.]+)', msg)
            if wait_str:
                wait_secs = parse_groq_wait_time(wait_str.group(1))
                st.warning(f"⏳ Rate limit hit. Retrying in {int(wait_secs)} seconds...")
                time.sleep(wait_secs)
                return synthesize_tts_file(text, voice, fmt)
            else:
                st.error(f"Rate limit hit: {msg}")
                st.stop()
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        st.error(f"TTS Error: {e}")
        st.stop()

def autoplay_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    with open(tmpfile_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        md = f"""
        <div class='audio-player'>
            <audio controls autoplay="true">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            <div style="margin-top:10px; text-align:center;">
                <img src="https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif" width="100%" style="border-radius:12px;" alt="waveform">
            </div>
        </div>
        """
        st.markdown(md, unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📄 Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = ""
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success("✅ Document uploaded!")

    st.markdown("### 🎙 Choose a Voice")
    emoji_voices = {
        "Arista-PlayAI": "🎤 Arista",
        "Atlas-PlayAI": "🧔 Atlas",
        "Basil-PlayAI": "🧙 Basil",
        "Briggs-PlayAI": "🔊 Briggs",
        "Calum-PlayAI": "🧑 Calum",
        "Celeste-PlayAI": "💃 Celeste",
        "Cheyenne-PlayAI": "👩 Cheyenne",
        "Chip-PlayAI": "🤠 Chip",
        "Cillian-PlayAI": "🇮🇪 Cillian",
        "Deedee-PlayAI": "🧓 Deedee",
        "Fritz-PlayAI": "🎧 Fritz",
        "Gail-PlayAI": "👵 Gail",
        "Indigo-PlayAI": "🦄 Indigo",
        "Mamaw-PlayAI": "👵 Mamaw",
        "Mason-PlayAI": "👦 Mason",
        "Mikail-PlayAI": "🧑‍💼 Mikail",
        "Mitch-PlayAI": "👨 Mitch",
        "Quinn-PlayAI": "👩 Quinn",
        "Thunder-PlayAI": "⚡ Thunder"
    }
    display_names = list(emoji_voices.values())
    selected_display = st.selectbox("Voice", display_names, index=10)
    voice = list(emoji_voices.keys())[display_names.index(selected_display)]

# ---- AUDIO INPUT ----
st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is None:
    html("""
    <div style='text-align:center; padding-top:10px;'>
        <div style="animation: float 2s ease-in-out infinite; font-size: 60px;">🎙️</div>
        <p style="color:gray; font-weight: bold;">Waiting for voice input...</p>
    </div>
    """, height=150)

if audio_bytes is not None:
    if not resume_text:
        st.warning("Please upload your document PDF first.")
        st.stop()

    with st.spinner("🔍 Transcribing..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)

    st.markdown("### 📝 Transcription")
    st.markdown(f"<div class='transcript-box'>{transcription}</div>", unsafe_allow_html=True)

    prompt = f"""
    You are a helpful voice assistant that answers spoken questions either based on a document (like a resume) or from general knowledge.

    Instructions:
    - If the question is about the uploaded document, answer as if *you are the person described* — use natural first-person tone, but *do not repeat the person's name*.
    - If the question is unrelated to the document, give a short, clear factual answer.

    Keep the response concise (under 150 words) and natural.

    Document:
    {resume_text}

    Question:
    {transcription}
    """

    with st.spinner("💡 Thinking..."):
        reply = generate_response_groq_direct(prompt)

    st.markdown("### 🤖 Assistant Says")
    st.markdown(f"<div style='font-size:22px; background-color:#ffe0f0; border-left: 6px solid #ff5e57; padding:15px; border-radius:10px;'>{reply}</div>", unsafe_allow_html=True)

    with st.spinner("🔈 Generating Voice..."):
        audio_response = synthesize_tts_file(reply, voice=voice, fmt="wav")

    autoplay_audio_bytes(audio_response)

st.caption("⚡ Powered by Groq, faster-whisper, Streamlit & 🤘 vibes.")
