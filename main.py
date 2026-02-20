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
import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def _sb_headers():
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

async def supabase_insert_memory(data: dict):
    headers = _sb_headers()
    if headers is None:
        raise RuntimeError("Supabase 未配置")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/memories"
    # Prefer: return=representation 会把插入后的行返回
    headers["Prefer"] = "return=representation"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=data)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase insert failed {r.status_code}: {r.text}")
        return r.json()

async def supabase_search_memories(query: str, limit: int = 10):
    headers = _sb_headers()
    if headers is None:
        raise RuntimeError("Supabase 未配置")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/memories"
    params = {
        "select": "id,title,content,category,importance,created_at",
        "or": f"(title.ilike.*{query}*,content.ilike.*{query}*)",
        "order": "created_at.desc",
        "limit": str(limit),
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase search failed {r.status_code}: {r.text}")
        return r.json()

async def supabase_recent_memories(limit: int = 10):
    headers = _sb_headers()
    if headers is None:
        raise RuntimeError("Supabase 未配置")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/memories"
    params = {
        "select": "id,title,content,category,importance,created_at",
        "order": "created_at.desc",
        "limit": str(limit),
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase recent failed {r.status_code}: {r.text}")
        return r.json()
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
    
    # tools/call
    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        arguments = params.get("arguments") or {}

        try:
            if name == "save_memory":
                title = arguments.get("title")
                content = arguments.get("content")
                category = arguments.get("category", "default")
                importance = int(arguments.get("importance", 3))

                if not title or not content:
                    return jsonrpc_error(_id, -32602, "title/content required")

                row = {
                    "title": title,
                    "content": content,
                    "category": category,
                    "importance": importance,
                }

                inserted = await supabase_insert_memory(row)

                return jsonrpc_result(
                    _id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"已写入 Supabase ✅\n插入结果: {inserted}",
                            }
                        ]
                    },
                )

            if name == "search_memory":
                q = arguments.get("query")
                if not q:
                    return jsonrpc_error(_id, -32602, "query required")

                items = await supabase_search_memories(q, limit=10)
                return jsonrpc_result(
                    _id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"检索到 {len(items)} 条\n{json.dumps(items, ensure_ascii=False, indent=2)}",
                            }
                        ]
                    },
                )

            if name == "get_recent_memories":
                limit = int(arguments.get("limit", 10))
                items = await supabase_recent_memories(limit=limit)
                return jsonrpc_result(
                    _id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"最近 {len(items)} 条\n{json.dumps(items, ensure_ascii=False, indent=2)}",
                            }
                        ]
                    },
                )

            return jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

        except Exception as e:
            # 把 supabase 的错误直接回给你，方便你定位
            return jsonrpc_result(
                _id,
                {
                    "content": [
                        {"type": "text", "text": f"工具执行失败 ❌\n{repr(e)}"}
                    ]
                },
            )

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
