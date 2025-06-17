import streamlit as st
st.set_page_config(page_title="Voice Bot with Groq + Hume", layout="centered")

import PyPDF2
import requests
import tempfile
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel

# --- API Keys from secrets ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HUME_API_KEY = st.secrets["HUME_API_KEY"]

# --- Constants ---
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3-70b-8192"  # or "llama-3-8b-8192"

# Load Whisper Model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")  # Use "cuda" if GPU is available

whisper_model = load_whisper_model()

st.title("🎙 Personalized Voice Bot (Groq + Hume AI)")

# --- Functions ---
def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return "\n".join(p.extract_text() for p in pdf_reader.pages if p.extract_text()).strip()

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        segments, _ = whisper_model.transcribe(tmpfile.name)
    return " ".join(segment.text for segment in segments)

def generate_response_groq(question, resume_text):
    prompt = f"""
You are a helpful assistant.

If the user's question is related to the resume below, answer as the person in the resume using first person.

If the question is general and not related to the resume, answer helpfully and factually as yourself.

Resume:
{resume_text}

Question:
{question}
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def synthesize_tts(text, description="friendly", fmt="wav"):
    headers = {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "utterances": [{"text": text, "description": description}],
        "format": {"type": fmt}
    }
    resp = requests.post("https://api.hume.ai/v0/tts/file", headers=headers, json=body)
    resp.raise_for_status()
    return resp.content

# --- UI ---
with st.sidebar:
    st.header("📄 Upload Resume (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = extract_pdf_text(pdf_file) if pdf_file else ""

st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    if resume_text:
        with st.spinner("🧠 Transcribing..."):
            transcription = transcribe_audio(audio_bytes)
        st.success(f"📝 Transcription: {transcription}")

        with st.spinner("💬 Generating reply..."):
            reply = generate_response_groq(transcription, resume_text)
        st.success("✅ Response ready!")
        st.markdown(f"**🗣 Answer:** {reply}")

        with st.spinner("🔊 Generating voice..."):
            audio_response = synthesize_tts(reply)
        st.audio(audio_response, format="audio/wav")
    else:
        st.warning("Please upload your resume first.")

st.caption("Powered by Groq (LLM), Hume AI (TTS), and faster-whisper (STT)")
