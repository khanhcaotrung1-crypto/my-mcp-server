import os
import json
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RikkaHub MCP Server (streamable_http)")

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

def jsonrpc_result(_id, result):
    return {"jsonrpc": "2.0", "id": _id, "result": result}

def jsonrpc_error(_id, code, message):
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}

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
                "serverInfo": {"name": "rikka-memory", "version": "0.1.0"},
            },
        )

    if method in ("tools/list", "list_tools"):
        return jsonrpc_result(_id, {"tools": TOOLS})

    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        arguments = params.get("arguments") or {}

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

        # TODO: 后续接入真实 supabase
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

@app.post("/mcp")
async def mcp_http(request: Request):
    # streamable_http：一次请求一次响应（application/json）
    payload = await request.json()
    resp = await handle_rpc(payload)
    return JSONResponse(resp, status_code=200)
