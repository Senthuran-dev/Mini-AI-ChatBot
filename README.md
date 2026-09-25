# 🤖 SeekAI

A conversational AI chatbot built with **Streamlit** and **Groq's OpenAI GPT-OSS 20B** model via **LlamaIndex** — now with **live web search**, so it answers current-events questions with up-to-date facts and shows its sources.

---

## ✨ Features

- 🎨 **Modern App UI** — features a sleek welcome screen, prompt cards, custom branding, and an unscrollable sidebar
- 🌐 **Live web search** — looks things up online before answering (Tavily, with automatic DuckDuckGo fallback)
- 🔗 **Clean Source Citations** — web sources are neatly hidden in collapsible expanders below each answer
- 📅 **Date-aware** — the model is told today's date on every request
- 💬 **Multi-turn conversation** — the bot remembers the chat history, and follow-ups like *"and his deputy?"* are understood
- ⚡ **Powered by Groq** — ultra-fast OpenAI GPT-OSS 20B inference
- 🛡️ **Honest fallbacks** — if search is down, you get a visible warning instead of a silently outdated answer
- 🔑 **Secure API keys** — loaded from `.env`, never hardcoded
- 🧪 **Unit tests** — the search and routing logic is covered by tests that need no internet

---

## 🧠 Why live search? (the "outdated answers" fix)

A language model only knows what it saw during training. Ask it *"Who is the CM of Tamil Nadu?"* and it answers from that old snapshot — and may even invent details. Since the model can't browse on its own, the app now does it for the model:

```text
User
 ↓
Streamlit UI
 ↓
Query Router (Groq)
 ↓
Needs fresh data?
├── No → Groq → Answer
└── Yes
     ↓
   Tavily Search
     ↓
   DuckDuckGo fallback
     ↓
   Search Results
     ↓
   Groq + Sources
     ↓
   Cited Answer
```

Search results exist only for that one request — they are never stored in the chat history.

---

## 📁 Project Structure

```
SeekAI/
├── assets/                 ← Contains UI assets (e.g., logo.png)
├── .venv/                  ← Python virtual environment
├── .env                    ← Your API keys (DO NOT commit this)
├── .env.example            ← Template for setting up .env
├── .gitignore              ← Keeps secrets and clutter out of Git
├── app.py                  ← Streamlit UI
├── chat_engine.py          ← Routing, prompt building, answer generation
├── web_search.py           ← Tavily / DuckDuckGo search, normalized results
├── tests/                  ← Unit tests (no network needed)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SeekAI
```

### 2. Create a virtual environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Copy the example env file:

```powershell
copy .env.example .env      # Windows
```
```bash
cp .env.example .env        # macOS / Linux
```

Then open `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_key_here     # optional but recommended
```

> 🔑 Get a **free** Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
> 🌐 Get a **free** Tavily key (1,000 credits/month, no card; each search here costs 1 credit) at [app.tavily.com](https://app.tavily.com).
> Without it the bot uses DuckDuckGo, which works but can be rate-limited.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

**Try it:** ask *"Who is the current CM of Tamil Nadu?"* — you should see a searched answer with a **Sources** dropdown underneath. Use the **🌐 Web Search** switch in the sidebar to compare with search turned off.

### 6. Customize your brand

Replace `assets/logo.png` with your own logo image to automatically update the favicon, sidebar branding, and welcome screen!

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Chat UI and web framework |
| [LlamaIndex](https://www.llamaindex.ai/) | LLM abstraction layer |
| [Groq](https://groq.com/) | Fast LLM inference API |
| [OpenAI GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) | Underlying language model |
| [Tavily](https://tavily.com/) | Web search built for LLM apps (primary) |
| [ddgs](https://pypi.org/project/ddgs/) | DuckDuckGo search, no key needed (fallback) |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure API key loading |

---

## ⚙️ Configuration

**Environment variables** (set in `.env`, or in Streamlit *secrets* when deploying):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Your Groq API key |
| `TAVILY_API_KEY` | ❌ | empty | Enables Tavily search; without it DuckDuckGo is used |
| `GROQ_MODEL` | ❌ | `openai/gpt-oss-20b` | Any chat model on Groq, e.g. `openai/gpt-oss-120b` for stronger answers |

**Code settings:**

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `temperature` | `get_llms()` in `app.py` | `0.3` | Answer creativity (0 = precise, 1 = random). Lowered from 0.7 so answers stick to the search results |
| `SYSTEM_PROMPT` | `chat_engine.py` | Helpful assistant | Bot's personality and instructions |
| `ALWAYS_SEARCH_RE` | `chat_engine.py` | see file | Words that always force a web search |
| `MAX_HISTORY` | `chat_engine.py` | `6` | How many recent messages are sent to the model |
| `max_results` | `web_search()` | `5` | Search results given to the model |

---

## 🧪 Running the Tests

No extra packages and no internet needed (the LLM and search are faked):

```bash
python -m unittest discover -s tests -v
```

---

## 🩺 Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Answers still look outdated | Make sure the **🌐 Live web search** switch is on. If you see a "search was unavailable" warning, add a `TAVILY_API_KEY` (DuckDuckGo may be rate-limiting you) |
| "This model is no longer available" | Groq retired the model. Set `GROQ_MODEL` in `.env` to a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models) |
| "Rate limit reached" | With live search on, every message makes two model calls (plan + answer). Wait a few seconds, or upgrade your Groq plan |
| "Groq rejected the API key" | Check `GROQ_API_KEY` in `.env` (no quotes or spaces) and restart the app |

---

## 🔒 Security Notes

- **Never commit `.env`** or `.streamlit/secrets.toml` — both are listed in `.gitignore`
- Use `.env.example` as a template to share what keys are needed
- Rotate your API keys immediately if they are ever exposed
- Web pages are untrusted text: the model is told to treat search results as information only and never to follow instructions found inside them

---

## 📄 License

This project is for educational purposes.
