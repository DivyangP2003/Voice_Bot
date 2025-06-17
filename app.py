import streamlit as st
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from gtts import gTTS
import openai
import tempfile

# ---- CONFIG ----
openai.api_key = st.secrets["api_keys"]["openai_key"]

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper()

# ---- UI ----
st.set_page_config(page_title="Free Voice Bot", layout="centered")
st.title("🎙️ Free Voice Bot (Whisper + GPT + gTTS)")

# ---- RECORD AUDIO ----
st.subheader("1. Record Your Question")
audio = st_audiorec()

if audio:
    st.audio(audio, format="audio/wav")

    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            f.flush()
            segments, _ = whisper_model.transcribe(f.name)
            transcription = " ".join(seg.text for seg in segments)

    st.success(f"📝 Transcribed: {transcription}")

    # ---- GPT-3.5 RESPONSE ----
    with st.spinner("Generating response..."):
        prompt = f"Answer this question helpfully: {transcription}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content.strip()

    st.success("✅ Response generated!")
    st.markdown(f"**🗣 Answer:** {reply}")

    # ---- TEXT TO SPEECH (gTTS) ----
    with st.spinner("Synthesizing speech..."):
        tts = gTTS(reply)
        audio_path = "response.mp3"
        tts.save(audio_path)
        audio_file = open(audio_path, "rb").read()
    st.audio(audio_file, format="audio/mp3")

st.caption("Uses Whisper for STT, GPT-3.5 for reply, and gTTS for speech output.")
