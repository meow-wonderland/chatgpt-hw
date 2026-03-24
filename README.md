# MyChat — HW01 Your own ChatGPT

A minimal but complete ChatGPT-like web app using the Anthropic Claude API.

## Features
- 🤖 Model selection (Claude Opus / Sonnet / Haiku)
- 📝 Custom system prompt
- 🎛️ Adjustable temperature & max tokens
- ⚡ Real-time streaming responses (SSE)
- 🧠 Configurable short-term memory (last N turns)

## Setup

### 1. Clone & install
```bash
git clone <your-repo>
cd chatgpt-hw/backend
pip install -r requirements.txt
```

### 2. Add your API key
```bash
cp ../.env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open frontend
Open `frontend/index.html` directly in your browser.  
*(No build step needed — it's plain HTML/JS)*

## Security
- API key is stored in `.env` (gitignored)
- Backend proxies all API calls — key never reaches the browser
- See `.gitignore` for what's excluded from git
