import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ===== OpenAI-compatible client (for LinkAPI + GLM) =====
from openai import OpenAI

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-2")

oai = None
if LLM_API_KEY:
    oai = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

app = FastAPI(title="RikkaHub MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TOOLS = [
    {
        "name": "save_memory",
        "description": "保存一条记忆到数据库",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "category": {"type": "string"},
                "importance": {"type": "integer"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "search_memory",
        "description": "搜索相关的记忆",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_memories",
        "description": "获取最近的记忆",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
]

SESSIONS: Dict[str, "asyncio.Queue[dict]"] = {}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "glm_embedding_ready": bool(oai),
    }

def sse_data(data) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"

async def sse_stream(session_id: str):
    yield sse_data({"type": "endpoint", "uri": f"message/{session_id}"})
    yield sse_data({"type": "ready", "ok": True})

    q = SESSIONS[session_id]

    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=25)
            yield sse_data(msg)
        except asyncio.TimeoutError:
            yield sse_data({"type": "ping", "t": datetime.now().isoformat()})

@app.get("/mcp")
async def mcp_sse():
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = asyncio.Queue()

    return StreamingResponse(
        sse_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

def jsonrpc_result(_id, result):
    return {"jsonrpc": "2.0", "id": _id, "result": result}

def jsonrpc_error(_id, code, message):
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}

# ===== GLM Embedding via LinkAPI =====
async def embed_text(text: str) -> list[float]:
    if not oai:
        raise RuntimeError("LLM_API_KEY 未配置")

    res = oai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding

async def handle_rpc(payload: dict):
    _id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return jsonrpc_result(
            _id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "rikka-memory", "version": "0.2.0"},
            },
        )

    if method in ("tools/list", "list_tools"):
        return jsonrpc_result(_id, {"tools": TOOLS})

    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "save_memory":
            title = arguments.get("title")
            content = arguments.get("content")
            if not (title and content):
                return jsonrpc_error(_id, -32602, "Missing title/content")

            emb = await embed_text(f"{title}\n{content}")
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Embedding length={len(emb)}，向量已生成（可用于写入Supabase）",
                        }
                    ]
                },
            )

        if name == "search_memory":
            query = arguments.get("query")
            if not query:
                return jsonrpc_error(_id, -32602, "Missing query")
            emb = await embed_text(query)
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Search embedding length={len(emb)}",
                        }
                    ]
                },
            )

        if name == "get_recent_memories":
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {"type": "text", "text": "Recent memories fetched (mock)."}
                    ]
                },
            )

    return jsonrpc_error(_id, -32601, f"Method not found: {method}")

@app.post("/mcp/message/{session_id}")
async def mcp_message(session_id: str, request: Request):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session")

    payload = await request.json()
    resp = await handle_rpc(payload)
    await SESSIONS[session_id].put(resp)
    return JSONResponse({"ok": True}, status_code=202)

@app.post("/message/{session_id}")
async def root_message(session_id: str, request: Request):
    return await mcp_message(session_id, request)

from urllib.parse import unquote

@app.post("/{weird:path}")
async def rikkahub_weird_endpoint_fix(weird: str, request: Request):
    decoded = unquote(weird)
    if not decoded.lstrip().startswith('{"uri"'):
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        obj = json.loads(decoded)
        uri = obj.get("uri", "")
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")
    if "message/" not in uri:
        raise HTTPException(status_code=404, detail="Not Found")
    session_id = uri.split("message/", 1)[1].split("/", 1)[0]
    return await mcp_message(session_id, request)
