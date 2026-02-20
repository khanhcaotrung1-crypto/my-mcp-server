import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict
from urllib.parse import unquote

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

# 每个 session 一个队列，用来 SSE 推送响应
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


def sse_data(data) -> str:
    # 最通用：只发 data:（不发 event:）
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def sse_keepalive_comment() -> str:
    # ✅ 代理最吃这一套：SSE 注释心跳，避免 idle/断开
    return ":\n\n"


async def sse_stream(session_id: str):
    # ✅ endpoint 双份：兼容 RikkaHub/ktor 的各种解析方式
    yield sse_data(f"message/{session_id}")  # 纯字符串（有人直接当 URL）
    yield sse_data({"type": "endpoint", "uri": f"message/{session_id}"})  # JSON（有人读 uri 字段）
    yield sse_data({"type": "ready", "ok": True})

    q = SESSIONS[session_id]

    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=10)
            yield sse_data(msg)
        except asyncio.TimeoutError:
            # ✅ 注释心跳，不会被客户端误当消息，但能保活
            yield sse_keepalive_comment()


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

    # initialize
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

        # 保底：没配环境变量也别把握手搞崩
        if not (SUPABASE_URL and SUPABASE_KEY):
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "Supabase 未配置。请在 Railway Variables 添加 SUPABASE_URL / SUPABASE_KEY",
                        }
                    ]
                },
            )

        # TODO：后续接入真实 supabase 存取逻辑
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


async def ensure_session(session_id: str) -> "asyncio.Queue[dict]":
    # ✅ 实例重启/偶发丢 session 时，自动补一个，别直接 404 卡死
    if session_id not in SESSIONS:
        SESSIONS[session_id] = asyncio.Queue()
    return SESSIONS[session_id]


@app.post("/mcp/message/{session_id}")
async def mcp_message(session_id: str, request: Request):
    q = await ensure_session(session_id)

    payload = await request.json()
    resp = await handle_rpc(payload)

    # ✅ 双保险：
    # 1) 同步 HTTP 直接返回 resp（ktor 很吃这一套）
    # 2) 同时推入 SSE 队列（兼容按 SSE 等响应的实现）
    await q.put(resp)
    return JSONResponse(resp, status_code=200)


@app.post("/message/{session_id}")
async def root_message(session_id: str, request: Request):
    return await mcp_message(session_id, request)


@app.post("/mcp/{rest_of_path:path}")
async def mcp_fallback(rest_of_path: str, request: Request):
    if not rest_of_path.startswith("message/"):
        raise HTTPException(status_code=404, detail="Not Found")
    session_id = rest_of_path.split("/", 1)[1]
    return await mcp_message(session_id, request)


@app.post("/{weird:path}")
async def rikkahub_weird_endpoint_fix(weird: str, request: Request):
    # 处理： /{"uri": "message/<id>"} 这种离谱路径
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
