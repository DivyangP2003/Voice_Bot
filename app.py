import streamlit as st
import PyPDF2
import requests
import tempfile
import base64
import re
import time
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel

# ---- CONFIG ----
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"

LLM_MODEL = "llama3-70b-8192"
TTS_MODEL = "playai-tts"

# Whisper model load
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper_model()

# UI Setup
st.set_page_config(page_title="🎙 Voice Bot", layout="centered")
st.title("🎙 Personalized Voice Bot (Groq Fast LLM + TTS)")

# --- PDF Handling ---
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# --- Audio Transcription ---
def transcribe_audio_faster_whisper(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        segments, _ = whisper_model.transcribe(tmpfile.name)
        return " ".join(segment.text for segment in segments)

# --- LLM Response ---
def generate_response_groq_direct(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.5
    }
    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"⚠️ Error generating response from Groq: {e}")
        st.stop()

# --- TTS with Groq ---
def synthesize_tts_file(text, voice="Mitch-PlayAI", fmt="wav"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": fmt
    }
    try:
        resp = requests.post(GROQ_TTS_URL, headers=headers, json=payload)
        if resp.status_code == 429:
            msg = resp.json().get('error', {}).get('message', '')
            st.warning(f"Rate limit hit. Message: {msg}")
            st.stop()
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        st.error(f"Unexpected TTS error: {e}")
        st.stop()

# --- Autoplay Audio ---
def autoplay_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    with open(tmpfile_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <audio controls autoplay="true">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """, unsafe_allow_html=True)

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("📄 Upload Document (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = ""
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success("✅ Document uploaded and extracted.")

# --- Sidebar: Voice selection ---
with st.sidebar:
    st.header("🗣 Voice Options")
    selected_voice = st.selectbox("Choose a TTS voice", options=[
        "Mitch-PlayAI", "Rachel-PlayAI", "Elliot-PlayAI"
    ], index=0)

# --- Main: Voice Input ---
st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    if not resume_text:
        st.warning("Please upload your PDF resume or document first.")
        st.stop()

    with st.spinner("Transcribing audio locally..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)
    st.success(f"📝 Transcribed Text: {transcription}")

    # --- Prompt for LLM ---
    prompt = f"""
You are a professional assistant helping a user describe their resume.
The resume text is below. When answering questions, speak from the user's perspective and include facts from the resume. If the user asks about projects, respond like: 'My projects include...' without referencing yourself as an assistant.

Resume:
{resume_text}

User Question: {transcription}
""".strip()

    with st.spinner("Generating response with Groq..."):
        reply = generate_response_groq_direct(prompt)

    st.success("✅ Response generated")

    with st.container():
        st.markdown("### 🗣 Response")
        st.markdown(f"<div style='font-size:20px; padding:10px;'>{reply}</div>", unsafe_allow_html=True)

    with st.spinner("🔊 Generating audio with Groq TTS..."):
        audio_response = synthesize_tts_file(reply, voice=selected_voice)
        autoplay_audio_bytes(audio_response)

st.caption("Powered by Groq LLM & TTS, Faster-Whisper STT")
