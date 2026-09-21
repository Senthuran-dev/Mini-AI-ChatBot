# 🤖 Mini AI Chatbot

A conversational AI chatbot built with **Streamlit** and **Groq's OpenAI GPT-OSS 20B** model via **LlamaIndex**. Supports full multi-turn conversation memory, error handling, and a clean chat UI.

---

## ✨ Features

- 💬 **Multi-turn conversation** — the bot remembers the full chat history
- ⚡ **Powered by Groq** — ultra-fast OpenAI GPT-OSS 20B inference
- 🔄 **Loading spinner** — visual feedback while the bot is thinking
- 🛡️ **Error handling** — graceful messages instead of crashes
- 🔑 **Secure API key** — loaded from `.env`, never hardcoded
- 🧠 **System prompt** — gives the bot a helpful assistant persona

---

## 📁 Project Structure

```
Chatbot/
├── .venv/               ← Python virtual environment
├── .env                 ← Your API key (DO NOT commit this)
├── .env.example         ← Template for setting up .env
├── .gitignore           ← Keeps secrets and clutter out of Git
├── app.py               ← Main chatbot application
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Chatbot
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and fill in your key:

```powershell
copy .env.example .env
```

Then open `.env` and replace the placeholder with your real key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get a **free** Groq API key at [console.groq.com/keys](https://console.groq.com/keys)

### 5. Run the app

```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Chat UI and web framework |
| [LlamaIndex](https://www.llamaindex.ai/) | LLM abstraction layer |
| [Groq](https://groq.com/) | Fast LLM inference API |
| [OpenAI GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) | Underlying language model |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure API key loading |

---

## ⚙️ Configuration

You can tweak the following in [`app.py`](./app.py):

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `model` | `get_llm()` | `openai/gpt-oss-20b` | Groq model to use |
| `temperature` | `get_llm()` | `0.7` | Creativity (0 = precise, 1 = random) |
| `SYSTEM_PROMPT` | top of file | Helpful assistant | Bot's personality and instructions |

---

## 🔒 Security Notes

- **Never commit `.env`** — it's listed in `.gitignore` for safety
- Use `.env.example` as a template to share what keys are needed
- Rotate your API key immediately if it is ever exposed

---

## 📄 License

This project is for educational purposes.
