import streamlit as st
st.set_page_config(page_title="🎙 Voice Bot", layout="centered")

import PyPDF2
import requests
import tempfile
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

def is_resume_related(question):
    prompt = f"""Classify whether the following question is related to personal details like resume, job history, education, skills, etc., or if it's a general knowledge question. Return only 'resume' or 'general'.\n\nQuestion: {question}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.1
    }

    response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
    response.raise_for_status()

    classification = response.json()['choices'][0]['message']['content'].strip().lower()
    return classification == "resume"

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

    response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
    response.raise_for_status()

    return response.json()['choices'][0]['message']['content'].strip()

def synthesize_tts_file(text, description=None, fmt="wav"):
    headers = {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "utterances": [{"text": text}],
        "format": {"type": fmt}
    }

    if description:
        payload["utterances"][0]["description"] = description

    try:
        resp = requests.post("https://api.hume.ai/v0/tts/file",  headers=headers, json=payload)
        resp.raise_for_status()
        return resp.content  # Raw audio bytes
    except requests.exceptions.HTTPError as err:
        st.error(f"TTS Error: {resp.status_code} - {resp.text}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected TTS error: {e}")
        st.stop()

# APP UI
with st.sidebar:
    st.header("📄 Upload Resume (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = ""
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success("Resume uploaded and extracted!")

st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")

    if not resume_text:
        st.warning("Please upload your resume PDF first.")
        st.stop()

    with st.spinner("Transcribing audio locally with faster-whisper..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)
    st.success(f"📝 Transcription: {transcription}")

    if is_resume_related(transcription):
        prompt = f"""You are the person described in the following resume text:\n\n{resume_text}\n\nAnswer the question briefly and concisely in first person.\n\nQuestion: {transcription}"""
    else:
        prompt = f"""Answer the following question in a concise and factual manner:\n\n{transcription}"""

    with st.spinner("Generating response..."):
        reply = generate_response_groq_direct(prompt)
    st.success("✅ Response generated")
    st.markdown(f"🗣 Response: {reply}")

    with st.spinner("Synthesizing speech with Hume AI..."):
        audio_response = synthesize_tts_file(reply, description="clear and natural tone", fmt="wav")
    st.audio(audio_response, format="audio/wav")

st.caption("Powered by Hume AI (TTS), faster-whisper (STT), and Groq (LLMs).")
