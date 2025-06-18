import streamlit as st
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from groq import Groq

# ---- CONFIG ----
st.set_page_config(page_title="🎙 Voice Bot", layout="centered")

HUME_API_KEY = st.secrets["HUME_API_KEY"]  # No longer used, but kept for backup
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# ---- WHISPER MODEL LOADING ----
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")  # Use "cuda" for GPU if available

whisper_model = load_whisper_model()

# ---- GROQ CLIENT FOR TTS ----
@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

groq_client = load_groq_client()

# ---- PAGE TITLE ----
st.title("🎙 Personalized Voice Bot (Groq LLM + TTS + Whisper)")

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
        "max_tokens": 150,
        "temperature": 0.5
    }

    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error("⚠️ Error generating response from Groq.")
        st.stop()

def synthesize_tts_file(text, voice="Fritz-PlayAI", fmt="wav"):
    try:
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format=fmt
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as f:
            response.write_to_file(f.name)
            f.seek(0)
            audio_bytes = f.read()
        return audio_bytes

    except Exception as e:
        st.error(f"Groq TTS Error: {e}")
        st.stop()

def autoplay_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    with open(tmpfile_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
              <source src="data:audio/wav;base64,{b64}" type="audio/wav">
              Your browser does not support the audio element.
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

    import os
    os.unlink(tmpfile_path)

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📄 Upload Document (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = ""
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success("✅ Document uploaded and extracted!")

    st.markdown("### 🎙 Choose TTS Voice")
    voice = st.selectbox("Voice", [
        "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI",
        "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI",
        "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI",
        "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI",
        "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
    ], index=10)

# ---- AUDIO INPUT ----
st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    if not resume_text:
        st.warning("Please upload your document PDF first.")
        st.stop()

    with st.spinner("🔍 Transcribing audio locally..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)
    st.success(f"📝 Transcription: {transcription}")

    prompt = f"""You are a voice assistant that can either:
    - Act as the person described in the uploaded document (resume, report, etc.)
    - Or answer general knowledge questions directly.

    First, determine if the question relates to the document content or is general.
    Then, respond accordingly:
    - If document-related: Answer as if you are the person described in the document.
    - If general: Provide a concise factual answer.

    Keep your final answer under 150 words.

    Document:\n\n{resume_text}\n
    Question: {transcription}
    """

    with st.spinner("💡 Generating response with Groq..."):
        reply = generate_response_groq_direct(prompt)

    st.success("✅ Response generated")

    with st.container():
        st.markdown("### 🗣 Response")
        st.markdown(f"<div style='font-size:20px; padding:10px;'>{reply}</div>", unsafe_allow_html=True)

    with st.spinner("🔈 Synthesizing speech with Groq TTS..."):
        audio_response = synthesize_tts_file(reply, voice=voice, fmt="wav")

    autoplay_audio_bytes(audio_response)

st.caption("⚡ Powered by Groq (LLM + TTS), faster-whisper (STT)")
