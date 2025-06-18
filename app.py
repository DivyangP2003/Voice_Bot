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
import os

# ---- CONFIG ----
st.set_page_config(page_title="🎙 Voice Bot", layout="centered")

# ---- CUSTOM CSS ----
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: transparent;
        }
        .main {
            background-color: #111 !important;
        }
        h1, h2, h3 {
            color: #e4e4e4 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .markdown-text-container {
            font-size: 16px;
            color: #ccc !important;
            line-height: 1.6;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stSelectbox, .stButton, .stFileUploader {
            border-radius: 10px !important;
        }
        .stAlert {
            border-left: 5px solid #3a3a8e !important;
            background-color: #222 !important;
            color: #ccc !important;
        }
        .custom-card {
            background-color: #222;
            color: #ddd;
            padding: 15px;
            border-left: 5px solid #3a3a8e;
            border-radius: 8px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)
# ---- SECRETS ----
HUME_API_KEY = st.secrets["HUME_API_KEY"]  # Optional/unused
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

# ---- LOAD WHISPER MODEL ----
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper_model()

# ---- GROQ CLIENT ----
@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

groq_client = load_groq_client()

# ---- TITLE ----
st.title("🎙 Personalized Voice Bot")
st.markdown("Ask questions by voice — your AI assistant will respond using document context or general knowledge.")

# ---- FUNCTIONS ----
def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def transcribe_audio_faster_whisper(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        segments, _ = whisper_model.transcribe(tmpfile.name)
        transcription = " ".join(segment.text for segment in segments)
    return transcription

def generate_response_groq_direct(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error("⚠ Error generating response from Groq.")
        st.stop()

def parse_groq_wait_time(wait_str):
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?'
    match = re.match(pattern, wait_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0.0
    return hours * 3600 + minutes * 60 + seconds

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
            wait_str_match = re.search(r'in ([\dhms\.]+)', msg)
            if wait_str_match:
                wait_str = wait_str_match.group(1)
                wait_secs = parse_groq_wait_time(wait_str)
                st.warning(f"Rate limit hit. Waiting for {int(wait_secs)} seconds before retrying...")
                time.sleep(wait_secs)
                return synthesize_tts_file(text, voice, fmt)
            else:
                st.error(f"Rate limit hit, but couldn't parse wait time: {msg}")
                st.stop()
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as err:
        st.error(f"TTS Error: {resp.status_code} - {resp.text}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected TTS error: {e}")
        st.stop()

def autoplay_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name
    with open(tmpfile_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        md = f"""
            <audio controls autoplay style="width: 100%; margin-top: 10px;">
              <source src="data:audio/wav;base64,{b64}" type="audio/wav">
              Your browser does not support the audio element.
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    os.unlink(tmpfile_path)

# ---- STATIC RESUME LOADING (FROM TXT FILE) ----
RESUME_TXT_PATH = "assets/data.txt"

@st.cache_data
def load_resume_text():
    with open(RESUME_TXT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

resume_text = load_resume_text()

# Sidebar update
with st.sidebar:
    st.markdown("✅ Using preloaded resume text file.")
    st.divider()


    # Voice is now hardcoded
    voice = "Mitch-PlayAI"

# ---- AUDIO INPUT ----
st.markdown("## 🎤 Ask Your Voice Bot")
st.markdown("Click the mic below to record your question.")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    if not resume_text:
        st.warning("Please upload your document PDF first.")
        st.stop()

    with st.spinner("🔍 Transcribing audio..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)

    st.markdown(f"""
        <div class='custom-card'>
        <b>📝 Transcription:</b><br>{transcription}
        </div>
    """, unsafe_allow_html=True)


    prompt = f"""
    You are a helpful and concise voice assistant. Answer the question naturally as if you are the person described in the uploaded resume.
    
    Instructions:
    - Use a first-person tone ONLY if the question relates to the document.
    - Keep the response **between 3 to 5 sentences**.
    - Finish your thoughts completely; avoid ending mid-sentence.
    - If the question is unrelated to the resume, answer factually and briefly.
    - Do NOT include the person's name or mention the resume.
    

    Document:
    {resume_text}

    Question:
    {transcription}
    """

    with st.spinner("💡 Generating response..."):
        reply = generate_response_groq_direct(prompt)

    st.markdown("## 🧠 Assistant’s Response")
    st.markdown(f"""
        <div class='custom-card'>
        {reply}
        </div>
    """, unsafe_allow_html=True)
    

    with st.spinner("🔈 Synthesizing speech..."):
        audio_response = synthesize_tts_file(reply, voice=voice, fmt="wav")

    autoplay_audio_bytes(audio_response)

st.markdown("---")
st.markdown("<center><sub>⚡ Powered by <b>Groq</b> (LLM + TTS), <b>Faster-Whisper</b> (STT)</sub></center>", unsafe_allow_html=True)
