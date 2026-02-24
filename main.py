import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

"""
RikkaHub MCP (Streamable HTTP) — minimal, stable FastAPI server

✅ Recommended in RikkaHub:
type = "streamable_http"
url  = "https://<your-domain>/mcp"

Env (Railway Variables):
- PORT (Railway sets)
- SUPABASE_URL
- SUPABASE_KEY   (service_role key recommended for server-side)

Notes:
- This version avoids SSE handshake issues and works like common public MCP servers
  (client POSTs JSON-RPC to /mcp, server replies JSON-RPC in the HTTP response).
"""

app = FastAPI(title="RikkaHub MCP Server (Streamable HTTP)")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()

# --- MCP tool definitions ---
# Some clients require `inputSchema` (camelCase). Mirror `input_schema` for compatibility.
TOOLS = [
    {
        "name": "save_memory",
        "description": "保存一条记忆到数据库（Supabase memories 表）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"},
                "content": {"type": "string", "description": "内容"},
                "category": {"type": "string", "description": "分类，可选"},
                "importance": {"type": "integer", "description": "重要性 0-5，可选"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "search_memory",
        "description": "按关键词搜索记忆（匹配 title/content）",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_memories",
        "description": "获取最近的记忆（按时间倒序）",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "返回条数，默认 10"}},
        },
    },
]

for t in TOOLS:
    t["input_schema"] = t["inputSchema"]


# ---- Helpers: JSON-RPC ----
def jsonrpc_result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}


def jsonrpc_error(_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": _id, "error": err}


def tool_text(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


# ---- Supabase REST helpers (no extra deps) ----
def _sb_headers(prefer: Optional[str] = None) -> Dict[str, str]:
    if not SUPABASE_KEY:
        return {}
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _sb_ok() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def supabase_insert_memory(title: str, content: str, category: Optional[str], importance: Optional[int]) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/memories"
    payload: Dict[str, Any] = {"title": title, "content": content}
    if category:
        payload["category"] = category
    if importance is not None:
        payload["importance"] = int(importance)

    r = requests.post(url, headers=_sb_headers(prefer="return=representation"), data=json.dumps(payload), timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase insert failed ({r.status_code}): {r.text}")
    data = r.json()
    return data[0] if isinstance(data, list) and data else {"ok": True}


def supabase_search_memories(query: str, limit: int = 10) -> Any:
    q = query.replace("%", "").strip()
    like = f"*{q}*"
    or_filter = f"(title.ilike.{like},content.ilike.{like})"
    url = f"{SUPABASE_URL}/rest/v1/memories?select=*&or={quote(or_filter)}&order=created_at.desc&limit={int(limit)}"
    r = requests.get(url, headers=_sb_headers(), timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase search failed ({r.status_code}): {r.text}")
    return r.json()


def supabase_recent_memories(limit: int = 10) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/memories?select=*&order=created_at.desc&limit={int(limit)}"
    r = requests.get(url, headers=_sb_headers(), timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase recent failed ({r.status_code}): {r.text}")
    return r.json()


# ---- Basic routes ----
@app.get("/")
def root():
    return {"status": "ok", "service": "mcp-streamable-http"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "supabase_configured": _sb_ok(),
    }


# ---- MCP Streamable HTTP endpoint ----
@app.post("/mcp")
async def mcp(request: Request):
    """Streamable HTTP: client POSTs JSON-RPC -> server replies JSON-RPC in HTTP response."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    _id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if not method:
        return JSONResponse(jsonrpc_error(_id, -32600, "Invalid Request: missing method"), status_code=200)

    if method == "initialize":
        return JSONResponse(
            jsonrpc_result(
                _id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rikka-memory", "version": "0.2.0"},
                },
            ),
            status_code=200,
        )

    if method in ("tools/list", "list_tools"):
        return JSONResponse(jsonrpc_result(_id, {"tools": TOOLS}), status_code=200)

    if method in ("tools/call", "call_tool"):
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name not in {"save_memory", "search_memory", "get_recent_memories"}:
            return JSONResponse(jsonrpc_error(_id, -32601, f"Unknown tool: {name}"), status_code=200)

        if not _sb_ok():
            return JSONResponse(
                jsonrpc_result(
                    _id,
                    tool_text("Supabase 未配置。请在 Railway Variables 添加 SUPABASE_URL / SUPABASE_KEY（建议 service_role key）"),
                ),
                status_code=200,
            )

        try:
            if name == "save_memory":
                title = str(arguments.get("title", "")).strip()
                content = str(arguments.get("content", "")).strip()
                if not title or not content:
                    return JSONResponse(jsonrpc_error(_id, -32602, "Missing required: title/content"), status_code=200)

                category = arguments.get("category")
                importance = arguments.get("importance")

                row = supabase_insert_memory(title, content, category, importance)
                return JSONResponse(
                    jsonrpc_result(
                        _id,
                        tool_text(f"✅ 已写入 memories：id={row.get('id','?')} title={row.get('title','')}")),
                    status_code=200,
                )

            if name == "search_memory":
                q = str(arguments.get("query", "")).strip()
                if not q:
                    return JSONResponse(jsonrpc_error(_id, -32602, "Missing required: query"), status_code=200)

                rows = supabase_search_memories(q, limit=int(arguments.get("limit", 10) or 10))
                preview = "\n".join([f"- {r.get('title','(no title)')}: {str(r.get('content',''))[:60]}" for r in rows[:10]])
                text = "🔎 搜索结果：\n" + (preview if preview else "(空)")
                return JSONResponse(jsonrpc_result(_id, tool_text(text)), status_code=200)

            if name == "get_recent_memories":
                limit = int(arguments.get("limit", 10) or 10)
                rows = supabase_recent_memories(limit=limit)
                preview = "\n".join([f"- {r.get('title','(no title)')}: {str(r.get('content',''))[:60]}" for r in rows[:limit]])
                text = "🕒 最近记忆：\n" + (preview if preview else "(空)")
                return JSONResponse(jsonrpc_result(_id, tool_text(text)), status_code=200)

        except Exception as e:
            return JSONResponse(jsonrpc_error(_id, -32000, "Tool execution failed", data=str(e)), status_code=200)

    return JSONResponse(jsonrpc_error(_id, -32601, f"Method not found: {method}"), status_code=200)
  
  @app.get("/debug/embedding_dim")
async def debug_embedding_dim():
    vec = await embed_text("测试一下维度")
    return {"dim": len(vec)}
