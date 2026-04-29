from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import math
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

from database import (
    init_db, create_session, get_sessions, get_session,
    update_session_name, delete_session, add_message, get_messages,
)

load_dotenv()
init_db()

app = FastAPI(title="MyChatv2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Available models ──────────────────────────────────────────────────────────
MODELS = [
    {"id": "auto",                          "name": "🤖 Auto Route"},
    {"id": "claude-sonnet-4-20250514",      "name": "Claude Sonnet 4"},
    {"id": "claude-haiku-4-5-20251001",     "name": "Claude Haiku 4.5"},
    {"id": "claude-sonnet-4-20250514",    "name": "Claude 3.5 Sonnet (Vision)"},
    {"id": "claude-3-haiku-20240307",       "name": "Claude 3 Haiku"},
]

@app.get("/api/models")
def list_models():
    return {"models": MODELS}


# ── Auto routing ──────────────────────────────────────────────────────────────
COMPLEX_KEYWORDS = [
    "analyze", "explain", "code", "write", "create", "debug",
    "compare", "design", "algorithm", "implement", "refactor",
    "分析", "解釋", "程式", "寫", "設計", "比較", "實作",
]

def auto_route(last_message: str, has_image: bool) -> str:
    """Pick the cheapest model that can handle the request."""
    if has_image:
        return "claude-sonnet-4-20250514"   # vision-capable

    msg_lower = last_message.lower()
    if len(last_message) > 200 or any(k in msg_lower for k in COMPLEX_KEYWORDS):
        return "claude-sonnet-4-20250514"

    return "claude-haiku-4-5-20251001"        # fast & cheap for simple queries


# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            "Returns the current date and time. "
            "Use when the user asks what time or date it is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, e.g. 'Asia/Taipei'. Defaults to Asia/Taipei.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Safely evaluates a mathematical expression. "
            "Use for arithmetic, algebra, or any numerical computation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2**10', 'sqrt(144)', '100 * 1.05'",
                }
            },
            "required": ["expression"],
        },
    },
]


def run_tool(name: str, inputs: dict) -> str:
    if name == "get_current_time":
        tz_name = inputs.get("timezone", "Asia/Taipei")
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(tz_name))
        except Exception:
            now = datetime.utcnow()
            tz_name = "UTC"
        return f"Current time in {tz_name}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"

    if name == "calculate":
        expr = inputs.get("expression", "")
        try:
            safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe_globals.update({"abs": abs, "round": round, "min": min, "max": max})
            result = eval(expr, {"__builtins__": {}}, safe_globals)  # noqa: S307
            return f"{expr} = {result}"
        except Exception as exc:
            return f"Error: {exc}"

    return f"Unknown tool '{name}'"


# ── Session endpoints ─────────────────────────────────────────────────────────
@app.get("/api/sessions")
def list_sessions_ep():
    return get_sessions()

@app.post("/api/sessions")
def new_session():
    sid = str(uuid.uuid4())
    name = f"New chat  {datetime.now().strftime('%m/%d %H:%M')}"
    return create_session(sid, name)

@app.get("/api/sessions/{session_id}/messages")
def get_session_msgs(session_id: str):
    return get_messages(session_id)

@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"ok": True}

@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, body: dict):
    update_session_name(session_id, body.get("name", "Chat"))
    return {"ok": True}


# ── Chat endpoint ─────────────────────────────────────────────────────────────
class ImageData(BaseModel):
    media_type: str   # e.g. "image/jpeg"
    data: str         # base64-encoded bytes

class Message(BaseModel):
    role: str
    content: str
    image: Optional[ImageData] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    model: str = "auto"
    system_prompt: Optional[str] = "You are a helpful assistant."
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    use_tools: Optional[bool] = True
    stream: Optional[bool] = True


def to_api_messages(messages: list[Message]) -> list[dict]:
    """Convert to Anthropic API message format, embedding images where needed."""
    result = []
    for m in messages:
        if m.image and m.role == "user":
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": m.image.media_type,
                            "data": m.image.data,
                        },
                    },
                    {"type": "text", "text": m.content or "What's in this image?"},
                ],
            })
        else:
            result.append({"role": m.role, "content": m.content})
    return result


@app.post("/api/chat")
async def chat(req: ChatRequest):
    last_msg  = req.messages[-1] if req.messages else None
    has_image = bool(last_msg and last_msg.image)
    last_text = (last_msg.content if last_msg else "") or ""

    # ── Pick model ────────────────────────────────────────────────────────────
    actual_model = req.model
    if req.model == "auto":
        actual_model = auto_route(last_text, has_image)

    api_msgs = to_api_messages(req.messages)
    system   = req.system_prompt or "You are a helpful assistant."
    tools    = TOOLS if req.use_tools else []

    # ── Persist user message ──────────────────────────────────────────────────
    if req.session_id:
        add_message(req.session_id, "user", last_text,
                    has_image=has_image, model_used=actual_model)
        session = get_session(req.session_id)
        if session and session["name"].startswith("New chat") and last_text:
            short = last_text[:45] + ("…" if len(last_text) > 45 else "")
            update_session_name(req.session_id, short)

    # ── Generator (SSE) ───────────────────────────────────────────────────────
    def generate():
        full_text = ""

        if tools:
            # Step 1 — non-streaming call so we can detect tool_use stop reason
            response = client.messages.create(
                model=actual_model,
                max_tokens=req.max_tokens,
                system=system,
                messages=api_msgs,
                tools=tools,
                temperature=req.temperature,
            )

            if response.stop_reason == "tool_use":
                # Execute every tool the model requested
                tool_results  = []
                tool_info_out = []

                for block in response.content:
                    if block.type == "tool_use":
                        result = run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                        tool_info_out.append({
                            "name":   block.name,
                            "input":  block.input,
                            "result": result,
                        })

                # Tell the frontend which tools ran
                yield f"data: {json.dumps({'tool_calls': tool_info_out})}\n\n"

                # Rebuild assistant content for the API
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type":  "tool_use",
                            "id":    block.id,
                            "name":  block.name,
                            "input": block.input,
                        })

                follow_up = api_msgs + [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user",      "content": tool_results},
                ]

                # Step 2 — stream the final answer
                with client.messages.stream(
                    model=actual_model,
                    max_tokens=req.max_tokens,
                    system=system,
                    messages=follow_up,
                    temperature=req.temperature,
                ) as s:
                    for text in s.text_stream:
                        full_text += text
                        yield f"data: {json.dumps({'text': text, 'model': actual_model})}\n\n"

            else:
                # No tool call — just emit the text we already have
                for block in response.content:
                    if hasattr(block, "text"):
                        full_text += block.text
                        yield f"data: {json.dumps({'text': block.text, 'model': actual_model})}\n\n"

        else:
            # Streaming without tools
            with client.messages.stream(
                model=actual_model,
                max_tokens=req.max_tokens,
                system=system,
                messages=api_msgs,
                temperature=req.temperature,
            ) as s:
                for text in s.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'text': text, 'model': actual_model})}\n\n"

        # ── Persist assistant reply ───────────────────────────────────────────
        if req.session_id and full_text:
            add_message(req.session_id, "assistant", full_text, model_used=actual_model)

        yield f"data: {json.dumps({'done': True, 'model': actual_model})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
