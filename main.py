import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote

# =========================================================
# RikkaHub MCP Server (streamable_http transport)
# - SSE at GET  /mcp
# - JSON-RPC POST /message/{session_id}
# - Compatible fallback paths for some clients
# - Supabase (PostgREST) storage + pgvector semantic search
# =========================================================

app = FastAPI(title="RikkaHub MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Env ----------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
# For third-party OpenAI-compatible providers, set e.g. https://api.linkapi.ai/v1
OPENAI_API_URL = (os.getenv("OPENAI_API_URL") or "https://api.openai.com/v1").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"  # default fallback

# (Optional) you can keep it for future, not used in this version
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or ""

# ---------- MCP tool definitions ----------
TOOLS = [
    {
        "name": "save_memory",
        "description": "保存一条记忆到数据库（同时写入 embedding 向量，便于语义检索）",
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
        "description": "语义搜索相关记忆（优先向量检索，失败则降级关键字检索）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_memories",
        "description": "获取最近的记忆",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        },
    },
]

# ---------- SSE sessions ----------
SESSIONS: Dict[str, "asyncio.Queue[dict]"] = {}

from fastapi import Request
from fastapi.responses import StreamingResponse

@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_entry(request: Request):
    # GET: 标准 streamable_http 方式，建立 SSE 流
    if request.method == "GET":
        return await mcp_sse()

    # POST: 一些客户端（尤其是 ktor-client）会误把 /mcp 当成 JSON-RPC 入口
    # 这时不能返回 SSE（会导致客户端一直等到超时，表现为“连不上”）
    try:
        payload = await request.json()
    except Exception:
        payload = None

    # 1) 如果它就是 JSON-RPC（含 method 字段），直接同步返回 JSON-RPC 响应
    if isinstance(payload, dict) and payload.get("method"):
        resp = await handle_rpc(payload)
        return JSONResponse(resp, status_code=200)

    # 2) 否则当作“握手”请求：创建 session，返回 endpoint 信息（非流式）
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = asyncio.Queue()
    return JSONResponse(
        {"type": "endpoint", "uri": f"message/{session_id}", "ok": True},
        status_code=200,
    )

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_api_url": OPENAI_API_URL,
        "embedding_model": EMBEDDING_MODEL,
    }


# ---------- SSE helpers ----------
def sse_data(data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def sse_stream(session_id: str):
    # RikkaHub handshake expectations
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


# Some clients will call /mcp/message/<id>
@app.post("/mcp/message/{session_id}")
async def mcp_message(session_id: str, request: Request):
    return await _handle_message(session_id, request)


# Some clients will call /message/<id>
@app.post("/message/{session_id}")
async def root_message(session_id: str, request: Request):
    return await _handle_message(session_id, request)


# Some clients (ktor-client) may POST to a weird JSON path like /{"uri":"message/<id>"}
@app.post("/{weird:path}")
async def weird_endpoint_fix(weird: str, request: Request):
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
    return await _handle_message(session_id, request)


# ---------- JSON-RPC core ----------
def jsonrpc_result(_id: Any, result: Any):
    return {"jsonrpc": "2.0", "id": _id, "result": result}


def jsonrpc_error(_id: Any, code: int, message: str, data: Any = None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": _id, "error": err}


async def _handle_message(session_id: str, request: Request):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    resp = await handle_rpc(payload)

    # Transport style: push response through SSE, HTTP returns quickly
    await SESSIONS[session_id].put(resp)
    return JSONResponse({"ok": True}, status_code=200)


# ---------- OpenAI-compatible embeddings ----------
def _embedding_endpoint() -> str:
    # If OPENAI_API_URL already endswith /v1, use /embeddings after it
    # Examples:
    # - https://api.openai.com/v1
    # - https://api.linkapi.ai/v1
    return f"{OPENAI_API_URL}/embeddings"


async def embed_text(text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    url = _embedding_endpoint()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"model": EMBEDDING_MODEL, "input": text}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=body)

    if r.status_code >= 400:
        # Keep provider's error text for debugging
        raise RuntimeError(f"Embeddings HTTP {r.status_code}: {r.text}")

    data = r.json()
    try:
        return data["data"][0]["embedding"]
    except Exception:
        raise RuntimeError(f"Bad embeddings response: {data}")


def _vec_to_pgvector_literal(vec: List[float]) -> str:
    # pgvector accepts: '[0.1,0.2,0.3]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ---------- Supabase helpers ----------
def _supabase_headers():
    if not (SUPABASE_URL and SUPABASE_KEY):
        raise RuntimeError("Supabase 未配置：缺少 SUPABASE_URL / SUPABASE_KEY")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def supabase_insert_memory(
    title: str,
    content: str,
    category: Optional[str] = None,
    importance: Optional[int] = None,
    embedding_vec: Optional[List[float]] = None,
):
    url = f"{SUPABASE_URL}/rest/v1/memories"
    payload: Dict[str, Any] = {
        "title": title,
        "content": content,
    }
    if category is not None:
        payload["category"] = category
    if importance is not None:
        payload["importance"] = importance
    if embedding_vec is not None:
        payload["embedding"] = _vec_to_pgvector_literal(embedding_vec)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_supabase_headers(), json=payload)

    if r.status_code >= 400:
        raise RuntimeError(f"Supabase insert failed {r.status_code}: {r.text}")

    # By default PostgREST may return empty body unless you add Prefer
    return True


async def supabase_recent_memories(limit: int = 10):
    url = f"{SUPABASE_URL}/rest/v1/memories"
    params = {
        "select": "id,title,content,category,importance,created_at",
        "order": "created_at.desc",
        "limit": str(max(1, min(limit, 50))),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase query failed {r.status_code}: {r.text}")
    return r.json()


async def supabase_keyword_search(query: str, k: int = 5):
    # basic fallback if vector RPC not available
    url = f"{SUPABASE_URL}/rest/v1/memories"
    # title/content ILIKE
    params = {
        "select": "id,title,content,category,importance,created_at",
        "or": f"(title.ilike.*{query}*,content.ilike.*{query}*)",
        "order": "created_at.desc",
        "limit": str(max(1, min(k, 20))),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase keyword search failed {r.status_code}: {r.text}")
    return r.json()


async def supabase_vector_search(query_embedding: List[float], k: int = 5):
    """
    Requires you to create a SQL RPC function in Supabase:

    -- 1) enable pgvector
    create extension if not exists vector;

    -- 2) embedding column should be vector(1024) for your current model
    alter table memories
      alter column embedding type vector(1024);

    -- 3) index (optional but recommended once you have lots of rows)
    create index if not exists memories_embedding_idx
      on memories using ivfflat (embedding vector_cosine_ops) with (lists = 100);

    -- 4) RPC function
    create or replace function match_memories(
      query_embedding vector(1024),
      match_count int default 5
    )
    returns table(
      id uuid,
      title text,
      content text,
      category text,
      importance int,
      created_at timestamptz,
      similarity float
    )
    language sql stable
    as $$
      select
        m.id,
        m.title,
        m.content,
        m.category,
        m.importance,
        m.created_at,
        1 - (m.embedding <=> query_embedding) as similarity
      from memories m
      where m.embedding is not null
      order by m.embedding <=> query_embedding
      limit match_count;
    $$;
    """
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_memories"
    payload = {
        "query_embedding": _vec_to_pgvector_literal(query_embedding),
        "match_count": max(1, min(k, 20)),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_supabase_headers(), json=payload)

    if r.status_code == 404:
        # function not created yet
        raise RuntimeError("Supabase RPC match_memories not found (请先在 Supabase 里创建 SQL function)")
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase vector search failed {r.status_code}: {r.text}")
    return r.json()


# ---------- Tool handlers ----------
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

        try:
            if name == "save_memory":
                title = (arguments.get("title") or "").strip()
                content = (arguments.get("content") or "").strip()
                category = arguments.get("category")
                importance = arguments.get("importance")

                if not title or not content:
                    return jsonrpc_error(_id, -32602, "title/content required")

                # 1) compute embedding (title + newline + content)
                emb = await embed_text(title + "\n" + content)

                # 2) insert to supabase
                await supabase_insert_memory(
                    title=title,
                    content=content,
                    category=category,
                    importance=importance,
                    embedding_vec=emb,
                )

                return jsonrpc_result(
                    _id,
                    {
                        "content": [
                            {"type": "text", "text": "已写入 memories（含 embedding）"}
                        ]
                    },
                )

            if name == "get_recent_memories":
                limit = int(arguments.get("limit") or 10)
                rows = await supabase_recent_memories(limit=limit)
                return jsonrpc_result(
                    _id,
                    {"content": [{"type": "text", "text": json.dumps(rows, ensure_ascii=False)}]},
                )

            if name == "search_memory":
                query = (arguments.get("query") or "").strip()
                k = int(arguments.get("k") or 5)
                if not query:
                    return jsonrpc_error(_id, -32602, "query required")

                # Try semantic search via RPC
                try:
                    q_emb = await embed_text(query)
                    rows = await supabase_vector_search(q_emb, k=k)
                except Exception:
                    # fallback keyword search
                    rows = await supabase_keyword_search(query, k=k)

                return jsonrpc_result(
                    _id,
                    {"content": [{"type": "text", "text": json.dumps(rows, ensure_ascii=False)}]},
                )

            return jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

        except Exception as e:
            return jsonrpc_error(_id, -32000, "Tool execution error", data=str(e))

    return jsonrpc_error(_id, -32601, f"Method not found: {method}")


# ---------- Debug endpoint ----------
@app.get("/debug/embedding_dim")
async def debug_embedding_dim():
    try:
        vec = await embed_text("测试一下维度")
        return {"dim": len(vec), "head": vec[:5], "model": EMBEDDING_MODEL, "api": OPENAI_API_URL}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

# ---------- Debug endpoints ----------
@app.get("/debug/embedding_dim")
async def debug_embedding_dim():
    try:
        vec = await embed_text("测试一下维度")
        return {"dim": len(vec), "head": vec[:5], "model": EMBEDDING_MODEL, "api": OPENAI_API_URL}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

@app.get("/debug/vector_search")
async def debug_vector_search(q: str = "测试", k: int = 5):
    """用 query 文本走一次：embedding -> Supabase RPC match_memories"""
    try:
        q_emb = await embed_text(q)
        rows = await supabase_vector_search(q_emb, k=k)
        return {"ok": True, "k": k, "query": q, "rows": rows}
    except Exception as e:
        return {"ok": False, "error": str(e), "type": e.__class__.__name__}

@app.get("/debug/sql_snippet")
def debug_sql_snippet():
    """把需要在 Supabase SQL Editor 里执行的 SQL 片段吐出来，方便复制。"""
    return {
        "note": "去 Supabase Dashboard -> SQL Editor 执行下面这些（按需挑选）",
        "sql": [
            "create extension if not exists vector;",
            "alter table memories alter column embedding type vector(1024);",
            "create index if not exists memories_embedding_idx on memories using ivfflat (embedding vector_cosine_ops) with (lists = 100);",
            'create or replace function match_memories(\n  query_embedding vector(1024),\n  match_count int default 5\n)\nreturns table(\n  id uuid,\n  title text,\n  content text,\n  category text,\n  importance int,\n  created_at timestamptz,\n  similarity float\n)\nlanguage sql stable\nas $$\n  select\n    m.id,\n    m.title,\n    m.content,\n    m.category,\n    m.importance,\n    m.created_at,\n    1 - (m.embedding <=> query_embedding) as similarity\n  from memories m\n  where m.embedding is not null\n  order by m.embedding <=> query_embedding\n  limit match_count;\n$$;',
        ],
    }

