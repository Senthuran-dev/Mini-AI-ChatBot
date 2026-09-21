from __future__ import annotations

import os
import random
import tiktoken
from dataclasses import asdict

def get_token_count(messages: list) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        text = " ".join([m.get("content", "") for m in messages if m.get("content")])
        return len(enc.encode(text))
    except Exception:
        return 0

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
DEFAULT_GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-20b")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SeekAI",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chat history in session state early so headers can use it
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(100, 999)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: #0b0d12;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #171923;
        border-right: 1px solid #292c36;
    }

    /* Main title */
    .hero h1 {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #FAFAFA;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.7;
        color: #A0AEC0;
    }

    /* Status pills */
    .status-row {
        display: flex;
        gap: 10px;
        margin: 15px 0 30px 0;
        flex-wrap: wrap;
    }

    .status-row span {
        background: #1a1d27;
        border: 1px solid #2b2f3a;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
        color: #e2e8f0;
    }

    /* Prompt buttons */
    div.stButton > button {
        border-radius: 12px;
        text-align: left;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 18px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Disable Sidebar Scrolling */
    [data-testid="stSidebar"] {
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        overflow: hidden !important;
    }
    [data-testid="stSidebarUserContent"] {
        overflow: hidden !important;
    }

    /* Chat message containers */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* User chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: transparent !important;
        border-left: 2px solid #5C95FF !important;
    }
    
    /* Assistant chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1E1E2E !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── LLM singletons — created once per (key, model), not on every message ─────
# The key is a function argument so that fixing a missing/wrong key takes
# effect immediately instead of returning a stale cached result.
@st.cache_resource
def get_llms(api_key: str, model: str, temperature: float = 0.3):
    answer_llm = Groq(model=model, api_key=api_key, temperature=temperature)  # factual, less rambling
    router_llm = Groq(model=model, api_key=api_key, temperature=0.0)  # must be deterministic
    return answer_llm, router_llm


answer_llm, router_llm = get_llms(GROQ_API_KEY, DEFAULT_GROQ_MODEL, 0.3) if GROQ_API_KEY else (None, None)


# ── Caching web searches ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def cached_web_search(query: str, tavily_api_key: str | None = None, news: bool = False):
    import web_search
    return web_search.web_search(query, tavily_api_key=tavily_api_key, news=news)


def render_message(msg: dict) -> None:
    """Show one chat message, plus its web sources when it used a search."""
    st.markdown(msg["content"])
    if msg.get("search_failed"):
        st.warning("🌐 Live web search was unavailable, so this answer may be out of date.")
    sources = msg.get("sources")
    if sources:
        with st.expander(f"🔗 Sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                title = s["title"].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                url = s["url"]
                st.markdown(f"{i}. [{title}]({url})")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    import base64
    sidebar_logo = ""
    try:
        with open("assets/logo.png", "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode("utf-8")
        sidebar_logo = f'<img src="data:image/png;base64,{b64_logo}" style="width: 120px; border-radius: 10px; margin-bottom: 10px;">'
    except Exception:
        sidebar_logo = "<h3 style='margin-bottom: 5px; color: #FAFAFA;'>🤖 SeekAI</h3>"
        
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 10px;">
            {sidebar_logo}
            <div style="color: #A0AEC0; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 5px;">Web Research AI</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    search_enabled = st.toggle(
        "🌐 Web Search",
        value=True,
        help="Look up current information online before answering."
    )
    
    with st.expander("ℹ️ System Info"):
        if TAVILY_API_KEY:
            st.caption("Search provider: Tavily")
        else:
            st.caption("Search provider: DuckDuckGo. Add a `TAVILY_API_KEY` for more reliable results.")
            
        selected_model = DEFAULT_GROQ_MODEL
        temperature = 0.3
        st.caption(f"🤖 Model: {selected_model}")

    # Re-initialize LLMs with the selected model
    answer_llm, router_llm = get_llms(GROQ_API_KEY, selected_model, temperature) if GROQ_API_KEY else (None, None)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("<div style='text-align: center; font-size: 0.85rem;'><a href='https://github.com/Senthuran-dev/SeekAI' style='color: #5C95FF; text-decoration: none;'>View Source on GitHub</a></div>", unsafe_allow_html=True)

# Warn if API key is missing
if not GROQ_API_KEY:
    st.error(
        "🔑 **Groq API key not found!** "
        "Create a `.env` file in the project folder with:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# (Session state already initialized at the top)

# ── Welcome Screen or Chat History ────────────────────────────────────────────
if not st.session_state.messages:
    # Empty state / Welcome screen
    import base64
    logo_html = ""
    try:
        with open("assets/logo.png", "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'<img src="data:image/png;base64,{b64_logo}" style="height: 48px; vertical-align: middle; margin-right: 12px; margin-bottom: 8px;">'
    except Exception:
        pass

    st.markdown(
        f"""
        <div class="hero">
            <h1>{logo_html}SeekAI</h1>
            <p>Research, explore, and learn with AI-powered web search.</p>
        </div>
        <div class="status-row">
            <span>🟢 Online</span>
            <span>🌐 Web Search</span>
            <span>⚡ Groq</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔎 Research\n\nExplain the latest AI developments", use_container_width=True):
            st.session_state.initial_prompt = "Explain the latest AI developments"
            st.rerun()
    with col2:
        if st.button("💻 Coding\n\nExplain REST APIs with an example", use_container_width=True):
            st.session_state.initial_prompt = "Explain REST APIs with an example"
            st.rerun()
    with col3:
        if st.button("📰 News\n\nWhat happened in AI this week?", use_container_width=True):
            st.session_state.initial_prompt = "What happened in AI this week?"
            st.rerun()
    with col4:
        if st.button("📚 Learn\n\nTeach me how RAG works", use_container_width=True):
            st.session_state.initial_prompt = "Teach me how RAG works"
            st.rerun()
else:
    # Display all past messages
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            render_message(message)

# React to new user input
prompt = st.chat_input("Ask me anything...")

# Check if a prompt button was clicked
if "initial_prompt" in st.session_state:
    prompt = st.session_state.initial_prompt
    del st.session_state.initial_prompt

if prompt:
    if len(prompt) > 4000:
        st.warning("Please keep your message under 4,000 characters.")
        st.stop()

    # Show user message immediately
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Search (if needed) + answer, with a spinner while we wait
    with st.chat_message("assistant", avatar="🤖"):
        if search_enabled:
            with st.status("🔍 Searching the web...", expanded=True) as status:
                reply_stream = chat_engine.generate_reply(
                    st.session_state.messages,
                    answer_llm,
                    router_llm,
                    search_enabled=search_enabled,
                    tavily_api_key=TAVILY_API_KEY,
                    searcher=cached_web_search,
                )
                if reply_stream.search_query:
                    status.update(label=f"Searched for: {reply_stream.search_query}", state="complete", expanded=False)
                else:
                    status.update(label="Thinking...", state="complete", expanded=False)
        else:
            with st.spinner("Thinking..."):
                reply_stream = chat_engine.generate_reply(
                    st.session_state.messages,
                    answer_llm,
                    router_llm,
                    search_enabled=search_enabled,
                    tavily_api_key=TAVILY_API_KEY,
                    searcher=cached_web_search,
                )
            
        # Stream the text output
        text = st.write_stream(reply_stream.stream)
        
        assistant_message = {
            "role": "assistant",
            "content": text,
            "sources": [asdict(s) for s in reply_stream.sources],
            "search_query": reply_stream.search_query,
            "provider": reply_stream.provider,
            "search_failed": reply_stream.search_failed,
        }
        
        # Render the sources underneath the streamed text
        msg_without_content = assistant_message.copy()
        msg_without_content["content"] = ""
        render_message(msg_without_content)

    # Store the reply (with its sources) so it re-renders correctly on the next run
    st.session_state.messages.append(assistant_message)
