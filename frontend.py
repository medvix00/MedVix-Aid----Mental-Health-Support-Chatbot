import os
import json
import av
import numpy as np
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from vosk import Model, KaldiRecognizer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gtts import gTTS

# ================= ENV =================
load_dotenv()
st.set_page_config(page_title="Mental Health Support Chatbot", layout="wide")

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #e8f5e9;
}

/* Chat messages (user + assistant) */
[data-testid="stChatMessage"] * {
    color: #000000 !important;
}

/* Input text */
textarea, input {
    color: #000000 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Buttons text */
button {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# ================= CONFIG =================
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LOGO_PATH = "medvix_logo.jpeg"
VOSK_MODEL_PATH = "vosk model"

# ================= LOAD VOSK =================
@st.cache_resource
def load_vosk_model():
    if not os.path.exists(VOSK_MODEL_PATH):
        st.error("Vosk model folder not found. Place 'vosk model' next to frontend.py")
        st.stop()
    return Model(VOSK_MODEL_PATH)

# ================= VECTORSTORE =================
@st.cache_resource
def get_vectorstore():
    return FAISS.load_local(
        DB_FAISS_PATH,
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        ),
        allow_dangerous_deserialization=True
    )

# ================= PROMPT =================
CUSTOM_PROMPT_TEMPLATE = """"
You are a compassionate mental health support assistant.
Your role is to provide brief, calm, and supportive mental health guidance.

Detect the user's language and respond in the same language.
Use the provided context only if it is relevant to mental health support.
If the context is insufficient, say "I don't know" gently.

RESPONSE RULES:
- Keep responses short (3–5 sentences maximum).
- Use simple, reassuring language.
- Focus on emotional validation and one practical coping suggestion.
- Do not provide medical diagnoses or prescribe medication.
- Do not replace a mental health professional.
- Avoid long explanations or detailed lists.
- If the user expresses severe distress or self-harm thoughts, briefly encourage seeking professional or trusted help.

User message:
{question}

Mental health context:
{context}

Brief supportive response:
"""

def get_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ================= GREETING =================
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "namaste"]

# ================= AUDIO PROCESSOR =================
class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = KaldiRecognizer(load_vosk_model(), 16000)
        self.text = ""

    def recv_queued(self, frames):
        for frame in frames:
            audio = frame.to_ndarray().astype(np.int16)
            if self.recognizer.AcceptWaveform(audio.tobytes()):
                result = json.loads(self.recognizer.Result())
                self.text = result.get("text", "")
        return frames[-1]

# ================= TEXT TO SPEECH =================
def text_to_speech(text):
    tts = gTTS(text=text)
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()

# ================= QUERY =================
def process_query(query):
    if is_greeting(query):
        return "Hello. I’m here to listen and support you. How are you feeling today?"

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY
        ),
        chain_type="stuff",
        retriever=get_vectorstore().as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": get_prompt()},
    )
    return qa_chain.invoke({"query": query})["result"]

# ================= MAIN =================
def main():
    # ---------- LOGO ----------
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)

    st.title("MedVix Aid – Mental Health Support Assistant")

    # ---------- SESSION STATE ----------
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"Chat 1": []}
        st.session_state.current_chat = "Chat 1"

    if "voice_active" not in st.session_state:
        st.session_state.voice_active = False

    if "voice_lock" not in st.session_state:
        st.session_state.voice_lock = False

    # ---------- SIDEBAR ----------
    st.sidebar.title("Chat History")

    if st.sidebar.button("New Chat"):
        chat_name = f"Chat {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[chat_name] = []
        st.session_state.current_chat = chat_name

    chat_names = list(st.session_state.conversations.keys())
    selected_chat = st.sidebar.radio(
        "Select a chat:",
        chat_names,
        index=chat_names.index(st.session_state.current_chat)
    )
    st.session_state.current_chat = selected_chat
    messages = st.session_state.conversations[selected_chat]

    # ---------- DISPLAY CHAT ----------
    for msg in messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # IMPORTANT FIX
    user_text = None   # <-- FIX: always initialize

    # ---------- VOICE BUTTON ----------
    if st.button("🎤 Speak here"):
        st.session_state.voice_active = True
        st.session_state.voice_lock = False

    # ---------- MIC ----------
    if st.session_state.voice_active:
        webrtc_ctx = webrtc_streamer(
            key="mental-health-mic",
            audio_processor_factory=VoskAudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
        )

        if webrtc_ctx and webrtc_ctx.audio_processor:
            spoken = webrtc_ctx.audio_processor.text.strip()
            if spoken and not st.session_state.voice_lock:
                user_text = spoken
                st.session_state.voice_lock = True
                st.session_state.voice_active = False
                webrtc_ctx.audio_processor.text = ""

    # ---------- TEXT INPUT ----------
    typed_text = st.chat_input("Type your thoughts...")
    if typed_text:
        user_text = typed_text

    # ---------- PROCESS INPUT ----------
    if user_text:
        messages.append({"role": "user", "content": user_text})
        st.chat_message("user").markdown(user_text)

        response = process_query(user_text)
        messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

        st.audio(text_to_speech(response), format="audio/mp3")

if __name__ == "__main__":
    main()
