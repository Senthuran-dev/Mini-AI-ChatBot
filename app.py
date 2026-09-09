import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

# ── Load environment variables from .env ─────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ── LLM singleton — created once, not on every message ───────────────────────
@st.cache_resource
def get_llm():
    if not GROQ_API_KEY:
        return None
    return Groq(
        model="llama3-8b-8192",   # Fix #2: correct model name (was "Llma3-8b-8192")
        api_key=GROQ_API_KEY,     # Fix #1: loaded from .env, not hardcoded
        temperature=0.7            # Fix #9: lowered from 1 to 0.7
    )

llm = get_llm()

# ── System prompt — gives the bot a helpful persona ──────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, friendly, and concise AI assistant. "
    "Answer clearly and accurately. If you don't know something, say so honestly."
)

# ── Chat function with full conversation history and error handling ────────────
def chat_qa(messages: list[dict]) -> str:
    """
    Send the full conversation history to the LLM and return its reply as a string.
    Fix #3: LLM is a singleton (not re-created here).
    Fix #4: Wrapped in try/except for graceful error handling.
    Fix #5: Response explicitly cast to string.
    Fix #8: Full message history passed so the LLM has memory.
    Fix #15: System prompt included in every request.
    """
    if llm is None:
        return "⚠️ **API key not found.** Please add `GROQ_API_KEY` to your `.env` file and restart the app."

    try:
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        ]
        for msg in messages:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            chat_messages.append(ChatMessage(role=role, content=msg["content"]))

        response = llm.chat(chat_messages)
        return str(response.message.content)  # Fix #5: always a plain string

    except Exception as e:
        return f"⚠️ **An error occurred:** {str(e)}"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("My AI 🤖 :green[Chatbot] ✨")

# Warn if API key is missing
if not GROQ_API_KEY:
    st.error(
        "🔑 **Groq API key not found!** "
        "Create a `.env` file in the project folder with:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# Fix #14: Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to new user input
if prompt := st.chat_input("Ask me anything!"):
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Fix #6: Show spinner while waiting for LLM response
    with st.spinner("Thinking..."):
        reply = chat_qa(st.session_state.messages)

    # Show and store assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
