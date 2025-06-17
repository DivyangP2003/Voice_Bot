import streamlit as st
st.set_page_config(page_title="Voice Bot with Groq + Hume", layout="centered")
import PyPDF2
import requests
import tempfile
from faster_whisper import WhisperModel
from st_audiorec import st_audiorec

# ---- CONFIG ----
HUME_API_KEY = "KaiGK9GN7dtQ6dDCv8WqpaMKEoPzQlT9lV66ebfVnPMmMkJL"
GROQ_API_KEY = "gsk_ag68gxHbo2ED943rDWoSWGdy3FYqxnh11TtH7kYRsUaoW7QUEmQ"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3-70b-8192"  # update if needed

# ---- SESSION STATE INIT ----
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ---- WHISPER MODEL ----
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper_model()

# ---- APP TITLE ----
st.title("🎙 Personalized Voice Bot (Groq + Hume AI)")

# ---- PDF RESUME UPLOAD ----
with st.sidebar:
    st.header("📄 Upload Resume (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    if pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        st.session_state.resume_text = text.strip()
        st.success("Resume uploaded and extracted!")

    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation_history = []
        st.success("Conversation reset.")

# ---- AUDIO RECORDING ----
st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")

    if st.session_state.resume_text:
        with st.spinner("Transcribing audio locally with faster-whisper..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio_bytes)
                tmpfile.flush()
                segments, _ = whisper_model.transcribe(tmpfile.name)
                transcription = " ".join(segment.text for segment in segments)
        st.success(f"📝 Transcription: {transcription}")

        # --- GENERATE RESPONSE FROM GROQ ---
        def generate_response_groq(question, resume_text):
            # Include resume and conversation history
            messages = [{"role": "system", "content": f"You are the person described in the following resume:\n{resume_text}"}]
            messages.extend(st.session_state.conversation_history)
            messages.append({"role": "user", "content": question})
        
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": MODEL,
                "messages": messages,
            }
        
            response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        
            # DEBUG: Show error content if it fails
            if response.status_code != 200:
                st.error(f"❌ API Error {response.status_code}")
                st.code(response.text, language="json")
                response.raise_for_status()
        
            reply = response.json()['choices'][0]['message']['content'].strip()
        
            # Save conversation history
            st.session_state.conversation_history.append({"role": "user", "content": question})
            st.session_state.conversation_history.append({"role": "assistant", "content": reply})
        
            return reply

        with st.spinner("Generating personalized response with Groq..."):
            reply = generate_response_groq(transcription, st.session_state.resume_text)
        st.success("✅ Response generated")
        st.markdown(f"**🗣 Response:** {reply}")

        # --- TTS FROM HUME ---
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
            return resp.content

        with st.spinner("Synthesizing speech with Hume AI..."):
            audio_response = synthesize_tts_file(reply, description="friendly and natural tone", fmt="wav")
        st.audio(audio_response, format="audio/wav")

    else:
        st.warning("Please upload your resume PDF first.")

# ---- SHOW CHAT HISTORY ----
if st.session_state.conversation_history:
    st.markdown("### 🗨️ Conversation History")
    for msg in st.session_state.conversation_history:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")

st.caption("Powered by Hume AI (TTS), faster-whisper (STT), and Groq (LLMs).")
