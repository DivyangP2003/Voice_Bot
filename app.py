
import streamlit as st
st.set_page_config(page_title="Voice Bot with Groq + Hume", layout="centered")
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel

# ---- CONFIG ----
HUME_API_KEY = "KaiGK9GN7dtQ6dDCv8WqpaMKEoPzQlT9lV66ebfVnPMmMkJL"
GROQ_API_KEY = "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"
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
    prompt = f"""You are the person described in the following resume text:\n\n{resume_text}\n\nAnswer this question in first person:\n\n{question}"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

def synthesize_tts_file(text, description=None, fmt="wav"):
    headers = {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "utterances": [{"text": text, **({"description": description} if description else {})}],
        "format": {"type": fmt}
    }
    resp = requests.post("https://api.hume.ai/v0/tts/file", headers=headers, json=body)
    resp.raise_for_status()
    return resp.content  # raw audio bytes

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

    if resume_text:
        with st.spinner("Transcribing audio locally with faster-whisper..."):
            transcription = transcribe_audio_faster_whisper(audio_bytes)
        st.success(f"📝 Transcription: {transcription}")

        with st.spinner("Generating personalized response with Groq..."):
            reply = generate_response_groq(transcription, resume_text)
        st.success("✅ Response generated")
        st.markdown(f"**🗣 Response:** {reply}")

        with st.spinner("Synthesizing speech with Hume AI..."):
            audio_response = synthesize_tts_file(reply, description="friendly conversational tone", fmt="wav")
        st.audio(audio_response, format="audio/wav")
    else:
        st.warning("Please upload your resume PDF first.")

st.caption("Powered by Hume AI (TTS), faster-whisper (STT), and Groq (LLMs).")
