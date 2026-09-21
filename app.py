from __future__ import annotations

import os
from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

import chat_engine

# ── Load settings from .env / Streamlit secrets ──────────────────────────────
load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    """Streamlit secrets first (cloud deploys), then environment / .env."""
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    return value or os.getenv(name) or default


GROQ_API_KEY = get_setting("GROQ_API_KEY")
TAVILY_API_KEY = get_setting("TAVILY_API_KEY")  # optional: better web search
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-20b")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ── LLM singletons — created once per (key, model), not on every message ─────
# The key is a function argument so that fixing a missing/wrong key takes
# effect immediately instead of returning a stale cached result.
@st.cache_resource
def get_llms(api_key: str, model: str):
    answer_llm = Groq(model=model, api_key=api_key, temperature=0.3)  # factual, less rambling
    router_llm = Groq(model=model, api_key=api_key, temperature=0.0)  # must be deterministic
    return answer_llm, router_llm


answer_llm, router_llm = get_llms(GROQ_API_KEY, GROQ_MODEL) if GROQ_API_KEY else (None, None)


def render_message(msg: dict) -> None:
    """Show one chat message, plus its web sources when it used a search."""
    st.markdown(msg["content"])
    if msg.get("search_failed"):
        st.warning("🌐 Live web search was unavailable, so this answer may be out of date.")
    sources = msg.get("sources")
    if sources:
        st.caption(f"🔎 Searched the web for: *{msg['search_query']}* · via {msg['provider']}")
        with st.expander(f"Sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                title = s["title"].replace("[", "(").replace("]", ")")
                url = s["url"].replace("(", "%28").replace(")", "%29")
                date = f" — {s['published']}" if s.get("published") else ""
                st.markdown(f"**[{i}]** [{title}]({url}){date}")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Mini AI 🤖 :green[Chatbot] ✨")

with st.sidebar:
    st.markdown("### 🤖 Mini AI Chatbot")
    st.markdown(f"Powered by **Groq** — `{GROQ_MODEL}`")
    st.markdown("---")
    search_enabled = st.toggle(
        "🌐 Live web search",
        value=True,
        help="Look up current information online before answering, so answers aren't limited "
             "to the model's (older) training data.",
    )
    if search_enabled:
        if TAVILY_API_KEY:
            st.caption("Search provider: Tavily")
        else:
            st.caption("Search provider: DuckDuckGo. Add a `TAVILY_API_KEY` for more reliable results.")
    st.markdown("---")
    if st.button("Clear Chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("[View Source on GitHub](https://github.com/Senthuran-dev/Mini-AI-ChatBot)")

# Warn if API key is missing
if not GROQ_API_KEY:
    st.error(
        "🔑 **Groq API key not found!** "
        "Create a `.env` file in the project folder with:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

# React to new user input
if prompt := st.chat_input("Ask me anything!"):
    if len(prompt) > 4000:
        st.warning("Please keep your message under 4,000 characters.")
        st.stop()

    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Search (if needed) + answer, with a spinner while we wait
    with st.chat_message("assistant"):
        with st.spinner("Searching the web & thinking..." if search_enabled else "Thinking..."):
            reply = chat_engine.generate_reply(
                st.session_state.messages,
                answer_llm,
                router_llm,
                search_enabled=search_enabled,
                tavily_api_key=TAVILY_API_KEY,
            )
        assistant_message = {
            "role": "assistant",
            "content": reply.text,
            "sources": [asdict(s) for s in reply.sources],
            "search_query": reply.search_query,
            "provider": reply.provider,
            "search_failed": reply.search_failed,
        }
        render_message(assistant_message)

    # Store the reply (with its sources) so it re-renders correctly on the next run
    st.session_state.messages.append(assistant_message)
