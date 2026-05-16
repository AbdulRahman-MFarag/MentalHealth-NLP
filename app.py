import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindEase – Mental Health Support",
    page_icon="🌿",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD MODEL, TOKENIZER, LABEL ENCODER
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('mental_health_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_assets()

# ─────────────────────────────────────────────
# GROQ SETUP
# ─────────────────────────────────────────────
GROQ_API_KEY = "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"  # ← replace this
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAXLEN = 256  # change to 100 if you used that in your notebook

CLASS_META = {
    "Normal":               {"icon": "✅", "risk": "low"},
    "Anxiety":              {"icon": "😟", "risk": "medium"},
    "Depression":           {"icon": "💙", "risk": "medium"},
    "Bipolar":              {"icon": "🔄", "risk": "medium"},
    "Stress":               {"icon": "😓", "risk": "medium"},
    "Personality disorder": {"icon": "⚠️", "risk": "high"},
    "Suicidal":             {"icon": "🚨", "risk": "critical"},
}

SYSTEM_PROMPT = """You are MindEase, a warm and caring friend who understands mental health well.
You listen with empathy, speak naturally and gently, and never use clinical or robotic language.
You never use bullet points. You write like a caring friend — comforting first, then offering gentle advice.
Keep responses to 3–4 short paragraphs."""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def classify(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    return label, confidence

def build_initial_prompt(user_message: str, classification: str) -> str:
    return f"""The user came to you and wrote: "{user_message}"

Our system detected their message reflects: {classification}.

Write them a heartfelt, conversational message. Be gentle and human.
Do not mention the classification label or use clinical language.
Offer comfort first, then naturally weave in gentle advice.
Keep it to 3–4 short paragraphs."""

def get_groq_response(messages: list) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Groq chat history (role/content pairs including system prompt)
if "groq_history" not in st.session_state:
    st.session_state.groq_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

if "current_label" not in st.session_state:
    st.session_state.current_label = ""

if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 MindEase")
    st.caption("A safe space to share how you feel.")
    st.divider()
    st.warning("⚠️ This is not a substitute for professional mental health care.")
    st.markdown("**If you are in crisis, please contact:**")
    st.markdown("- 🆘 Emergency: 911 or local equivalent")
    st.markdown("- 💬 Crisis Text Line: Text HOME to 741741")
    st.markdown("- 🌍 International: findahelpline.com")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.groq_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.session_state.current_label = ""
        st.session_state.current_confidence = 0
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌿 MindEase")
st.caption("Share how you're feeling — MindEase will listen and offer support. This is not a replacement for professional help.")
st.divider()

# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌿"):
        label = msg.get("label", "")
        confidence = msg.get("confidence", 0)
        if label:
            meta = CLASS_META.get(label, {"icon": "✅", "risk": "low"})
            st.caption(f"{meta['icon']} Detected: **{label}** · {confidence:.0%} confidence")
        st.write(msg["content"])

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("How are you feeling right now?")

if user_input and user_input.strip():

    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("MindEase is thinking..."):

            # First message: classify + build initial prompt
            if len(st.session_state.messages) == 1:
                label, confidence = classify(user_input)
                st.session_state.current_label = label
                st.session_state.current_confidence = confidence

                meta = CLASS_META.get(label, {"icon": "✅", "risk": "low"})

                if meta["risk"] == "critical":
                    st.error("🚨 Your message contains signs of serious distress. Please reach out to a professional or crisis line immediately.")
                elif meta["risk"] == "high":
                    st.warning("⚠️ Your message suggests significant distress. Consider speaking with a mental health professional.")

                st.caption(f"{meta['icon']} Detected: **{label}** · {confidence:.0%} confidence")

                user_prompt = build_initial_prompt(user_input, label)

            else:
                # Continue naturally as a friend
                label = ""
                confidence = 0
                user_prompt = user_input

            # Add user message to Groq history and get response
            st.session_state.groq_history.append({"role": "user", "content": user_prompt})

            try:
                reply = get_groq_response(st.session_state.groq_history)
                st.session_state.groq_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                reply = "I'm having trouble connecting right now. Please try again in a moment."
                st.error(f"Debug error: {e}")

            st.write(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "label": label,
        "confidence": confidence
    })
