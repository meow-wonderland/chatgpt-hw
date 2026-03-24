from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS — allow your frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Request schema ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    system_prompt: Optional[str] = "You are a helpful assistant."
    messages: list[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = True

# ── Available models endpoint ────────────────────────────────────────────────
@app.get("/api/models")
def get_models():
    return {
        "models": [
            {"id": "claude-opus-4-5",            "name": "Claude Opus 4.5"},
            {"id": "claude-sonnet-4-20250514",   "name": "Claude Sonnet 4"},
            {"id": "claude-haiku-4-5-20251001",  "name": "Claude Haiku 4.5"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-haiku-20240307",    "name": "Claude 3 Haiku"},
        ]
    }

# ── Chat endpoint (supports streaming) ──────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        def generate():
            with client.messages.stream(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system_prompt or "You are a helpful assistant.",
                messages=messages,
                temperature=req.temperature,
            ) as stream:
                for text in stream.text_stream:
                    # Server-Sent Events format
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    else:
        response = client.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            system=req.system_prompt or "You are a helpful assistant.",
            messages=messages,
            temperature=req.temperature,
        )
        return {"content": response.content[0].text}
