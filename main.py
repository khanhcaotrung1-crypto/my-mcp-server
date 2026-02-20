import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RikkaHub MCP Server")

@app.post("/mcp/{rest_of_path:path}")
async def mcp_fallback(rest_of_path: str, request: Request):
    # 只处理 message/<id>
    if not rest_of_path.startswith("message/"):
        raise HTTPException(status_code=404, detail="Not Found")
    session_id = rest_of_path.split("/", 1)[1]
    return await mcp_message(session_id, request)
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
        "input_schema": {
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
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_memories",
        "description": "获取最近的记忆",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
]

# 每个 SSE 连接一个队列，用来推送 JSON-RPC 响应
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
    }


def sse(event: str, data: Any) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

async def sse_stream(session_id: str):
    # 立刻发 endpoint，RikkaHub 就靠这个完成握手
    yield sse("endpoint", f"message/{session_id}")
    yield sse("ready", {"ok": True})

    q = SESSIONS[session_id]

    # 保活 + 推送消息
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=25)
            yield sse("message", msg)
        except asyncio.TimeoutError:
            yield sse("ping", {"t": datetime.now().isoformat()})


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


async def handle_rpc(payload: dict):
    _id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    # MCP 常见：initialize
    if method == "initialize":
        return jsonrpc_result(
            _id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "rikka-memory", "version": "0.1.0"},
            },
        )

    # tools/list
    if method in ("tools/list", "list_tools"):
        return jsonrpc_result(_id, {"tools": TOOLS})

    # tools/call
    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        arguments = params.get("arguments") or {}

        # 先做一个“保底返回”，避免没配数据库就把连接搞崩
        if not (SUPABASE_URL and SUPABASE_KEY):
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "Supabase 未配置，先在 Railway Variables 添加 SUPABASE_URL / SUPABASE_KEY",
                        }
                    ]
                },
            )

        # 这里你后续再接入真实 supabase 存取逻辑
        return jsonrpc_result(
            _id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": f"已收到工具调用 {name}，arguments={arguments}",
                    }
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

    # 按 SSE transport 习惯：HTTP 端返回 202，真正响应走 SSE message 推回去
    await SESSIONS[session_id].put(resp)
    return JSONResponse({"ok": True}, status_code=202)

@app.post("/message/{session_id}")
async def root_message(session_id: str, request: Request):
    return await mcp_message(session_id, request)
from urllib.parse import unquote

@app.post("/{weird:path}")
async def rikkahub_weird_endpoint_fix(weird: str, request: Request):
    # Railway/HTTP logs 里看到的： /{"uri": "message/<id>"}
    decoded = unquote(weird)

    # 只处理这种 JSON 形态的“奇葩路径”
    if not decoded.lstrip().startswith('{"uri"'):
        raise HTTPException(status_code=404, detail="Not Found")

    try:
        obj = json.loads(decoded)
        uri = obj.get("uri", "")
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")

    # uri 可能是 "message/<id>" 或 "/mcp/message/<id>"
    if "message/" not in uri:
        raise HTTPException(status_code=404, detail="Not Found")

    session_id = uri.split("message/", 1)[1].split("/", 1)[0]
    return await mcp_message(session_id, request)
