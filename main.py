import os
import json
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote

# =========================================================
# RikkaHub MCP Server（精简版：仅保留推送功能）
# =========================================================

app = FastAPI(title="RikkaHub MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SH_TZ = timezone(timedelta(hours=8))

# ---------- Env ----------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN") or ""
CRON_SECRET = os.getenv("CRON_SECRET", "").strip()

# ---------- 只保留 2 个工具 ----------
TOOLS = [
    {
        "name": "pushplus_notify",
        "description": "推送一条消息到念念的手机",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "schedule_pushplus",
        "description": "创建定时推送任务。run_at 为 ISO 时间字符串（例如 2026-02-2508:30:00+08:00）。repeat可选：none/daily/weekly/hourly。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "run_at": {"type": "string"},
                "repeat": {
                    "type": "string",
                    "enum": ["none", "hourly", "daily", "weekly"],
                    "default": "none",
                },
            },
            "required": ["title", "content", "run_at"],
        },
    },
]

# ---------- Supabase helpers ----------
def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

# ---------- SSE sessions ----------
SESSIONS: Dict[str, asyncio.Queue] = {}


def sse_data(data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def sse_stream(session_id: str):
    yieldsse_data({"type": "endpoint", "uri": f"message/{session_id}"})
    yield sse_data({"type": "ready", "ok": True})
    q = SESSIONS[session_id]
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=25)
            yield sse_data(msg)
        except asyncio.TimeoutError:
            yield sse_data({"type": "ping", "t": datetime.now().isoformat()})


@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_entry(request: Request):
    if request.method == "GET":
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = asyncio.Queue()
        return StreamingResponse(
            sse_stream(session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    try:
        payload = await request.json()
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload.get("method"):
        resp = await handle_rpc(payload)
        if resp is None:
            from fastapi.responses import Response
            return Response(status_code=204)
        return JSONResponse(resp, status_code=200)
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = asyncio.Queue()
    return JSONResponse({"type": "endpoint", "uri": f"message/{session_id}", "ok": True})


@app.post("/mcp/message/{session_id}")
@app.post("/message/{session_id}")
async def handle_message(session_id: str, request: Request):
    if session_id not in SESSIONS:SESSIONS[session_id] = asyncio.Queue()
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    resp = await handle_rpc(payload)
    if resp is None:
        return JSONResponse({"ok": True})
    await SESSIONS[session_id].put(resp)
    return JSONResponse({"ok": True})


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "tools": [t["name"] for t in TOOLS]}


# ---------- JSON-RPC ----------
def jsonrpc_result(_id, result):
    return {"jsonrpc": "2.0", "id": _id, "result": result}

def jsonrpc_error(_id, code, message):
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}


async def handle_rpc(payload: dict):
    _id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return jsonrpc_result(_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "rikka-push", "version": "0.3.0"},
        })

    if method in ("tools/list", "list_tools"):
        return jsonrpc_result(_id, {"tools": TOOLS})

    if method is None or (isinstance(method, str) and method.startswith("notifications/")):
        return None

    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        args = params.get("arguments") or {}

        try:
            if name == "pushplus_notify":
                title = (args.get("title") or "").strip()
                content = (args.get("content") or "").strip()
                if not content:
                    return jsonrpc_error(_id, -32602, "content required")
                result = await pushplus_notify(title=title or "桑梨", content=content)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "schedule_pushplus":
                title = (args.get("title") or "").strip()
                content = (args.get("content") or "").strip()
                run_at = (args.get("run_at") or "").strip()
                repeat = (args.get("repeat") or "none").strip().lower()
                if not title or not content or not run_at:
                    return jsonrpc_error(_id, -32602, "title/content/run_at required")
                job = await create_push_schedule(title=title, content=content, run_at=run_at, repeat=repeat)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(job, ensure_ascii=False)}]})

            return jsonrpc_error(_id, -32601, f"Unknown tool: {name}")
        except Exception as e:
            return jsonrpc_error(_id, -32000, f"Error: {e}")

    return jsonrpc_error(_id, -32601, f"Method not found: {method}")


# ---------- PushPlus ----------
async def
