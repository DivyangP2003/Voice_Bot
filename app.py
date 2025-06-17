
import streamlit as st
st.set_page_config(page_title="Voice Bot with Groq + Hume", layout="centered")
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel

# ---- CONFIG ----

import streamlit as st
st.set_page_config(page_title="Voice Bot with Groq + Hume", layout="centered")
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel

# ---- CONFIG ----
HUME_API_KEY = st.secrets["HUME_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Load Whisper Model (loads once)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")  # Use "cuda" for GPU if available

whisper_model = load_whisper_model()

st.title("🎙 Personalized Voice Bot (Groq + Hume AI)")

# FUNCTIONS
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

def generate_response_groq(question, resume_text):
    prompt = f"""You are the person described in the following resume text:\n\
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Load Whisper Model (loads once)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")  # Use "cuda" for GPU if available

whisper_model = load_whisper_model()

st.title("🎙 Personalized Voice Bot (Groq + Hume AI)")

# FUNCTIONS
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

def generate_response_groq(question, resume_text):
    prompt = f"""You are the person described in the following resume text:\n\
