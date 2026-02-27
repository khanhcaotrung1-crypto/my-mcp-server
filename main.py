import os
import json
import asyncio
import uuid
import re
from datetime import datetime, timezone, timedelta
import datetime as dt
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
AMAP_KEY = os.getenv("AMAP_KEY") or ""
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN") or ""

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

{
    "name": "amap_geocode",
    "description": "把地址转换为经纬度（高德地理编码）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "address": {"type": "string", "description": "详细地址/地点名"},
            "city": {"type": "string", "description": "可选，城市（中文名或adcode）"}
        },
        "required": ["address"]
    }
},
{
    "name": "amap_reverse_geocode",
    "description": "把经纬度转换为地址（高德逆地理编码）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "纬度"},
            "lng": {"type": "number", "description": "经度"},
            "radius": {"type": "integer", "description": "可选，搜索半径，米", "default": 200}
        },
        "required": ["lat", "lng"]
    }
},
{
    "name": "amap_weather",
    "description": "查询天气（高德天气查询）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市adcode或中文名（建议adcode）"},
            "extensions": {"type": "string", "description": "base=实况，all=预报", "enum": ["base", "all"], "default": "base"}
        },
        "required": ["city"]
    }
},
{
    "name": "amap_poi_around",
    "description": "附近搜索POI（高德周边搜索）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "纬度"},
            "lng": {"type": "number", "description": "经度"},
            "keywords": {"type": "string", "description": "关键词，如：咖啡/便利店/地铁站"},
            "radius": {"type": "integer", "description": "半径米，默认1000", "default": 1000},
            "types": {"type": "string", "description": "可选，POI类型代码（高德types）"},
            "page": {"type": "integer", "default": 1},
            "offset": {"type": "integer", "default": 10}
        },
        "required": ["lat", "lng"]
    }
},
{
    "name": "amap_route_driving",
    "description": "驾车路线规划（高德驾车路径规划）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "origin_lat": {"type": "number"},
            "origin_lng": {"type": "number"},
            "dest_lat": {"type": "number"},
            "dest_lng": {"type": "number"},
            "strategy": {"type": "integer", "description": "路线策略，默认0（速度优先）", "default": 0}
        },
        "required": ["origin_lat", "origin_lng", "dest_lat", "dest_lng"]
    }
},
{
    "name": "pushplus_notify",
    "description": "推送一条消息到PushPlus（需要PUSHPLUS_TOKEN）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "template": {"type": "string", "enum": ["txt", "html", "markdown", "json"], "default": "txt"}
        },
        "required": ["title", "content"]
    }
},
{
    "name": "schedule_pushplus",
    "description": "创建/更新一个 PushPlus 定时推送任务。run_at 为 ISO 时间字符串（例如 2026-02-25 08:30:00+08:00 或 2026-02-25T08:30:00+08:00）。repeat 可选：none/daily/weekly/hourly。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "run_at": {"type": "string"},
            "repeat": {"type": "string", "enum": ["none", "hourly", "daily", "weekly"], "default": "none"},
            "template": {"type": "string", "enum": ["txt", "html", "markdown", "json"], "default": "txt"}
        },
        "required": ["title", "content", "run_at"]
    }
},
{
    "name": "list_pushplus_schedules",
    "description": "列出 PushPlus 定时推送任务。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20}
        }
    }
},
{
    "name": "cancel_pushplus_schedule",
    "description": "取消（禁用）一个 PushPlus 定时推送任务。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        },
        "required": ["id"]
    }
},
{
    "name": "run_due_pushplus",
    "description": "立即执行所有到期的 PushPlus 任务（通常给 /cron/tick 用）。",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
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


# 兼容别名：旧代码里用的是 supabase_headers()
def supabase_headers():
    return _supabase_headers()


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


            
            # -----------------
            # AMap tools
            # -----------------
            if name == "amap_geocode":
                address = (arguments.get("address") or "").strip()
                city = (arguments.get("city") or "").strip() or None
                if not address:
                    return jsonrpc_error(_id, -32602, "address required")
                data = await amap_geocode(address=address, city=city)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "amap_reverse_geocode":
                # allow either "location" ("lng,lat") or ("lng","lat")
                location = (arguments.get("location") or "").strip()
                if not location:
                    lng = arguments.get("lng")
                    lat = arguments.get("lat")
                    if lng is not None and lat is not None:
                        location = f"{lng},{lat}"
                if not location:
                    return jsonrpc_error(_id, -32602, "location required (lng,lat) or lng+lat")
                radius = int(arguments.get("radius") or 1000)
                extensions = (arguments.get("extensions") or "base").strip() or "base"
                data = await amap_reverse_geocode(location=location, radius=radius, extensions=extensions)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "amap_weather":
                city = (arguments.get("city") or "").strip()
                if not city:
                    return jsonrpc_error(_id, -32602, "city (adcode or city name) required")
                extensions = (arguments.get("extensions") or "base").strip() or "base"
                data = await amap_weather(city=city, extensions=extensions)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "amap_poi_around":
                # allow either "location" ("lng,lat") or ("lng","lat")
                location = (arguments.get("location") or "").strip()
                if not location:
                    lng = arguments.get("lng")
                    lat = arguments.get("lat")
                    if lng is not None and lat is not None:
                        location = f"{lng},{lat}"
                if not location:
                    return jsonrpc_error(_id, -32602, "location or lng+lat required")
                keywords = (arguments.get("keywords") or "").strip() or None
                types = (arguments.get("types") or "").strip() or None
                radius = int(arguments.get("radius") or 3000)
                data = await amap_poi_around(location=location, keywords=keywords, types=types, radius=radius)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "amap_route_driving":
                # allow either origin/destination ("lng,lat") or (origin_lng/origin_lat + destination_lng/destination_lat)
                origin = (arguments.get("origin") or "").strip()
                destination = (arguments.get("destination") or "").strip()
                if not origin:
                    olng = arguments.get("origin_lng")
                    olat = arguments.get("origin_lat")
                    if olng is not None and olat is not None:
                        origin = f"{olng},{olat}"
                if not destination:
                    dlng = arguments.get("destination_lng")
                    dlat = arguments.get("destination_lat")
                    if dlng is not None and dlat is not None:
                        destination = f"{dlng},{dlat}"
                if not origin or not destination:
                    return jsonrpc_error(_id, -32602, "origin/destination required (lng,lat) or origin_* + destination_*")
                data = await amap_route_driving(origin=origin, destination=destination)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "pushplus_notify":
                title = (arguments.get("title") or "RikkaHub 通知").strip()
                content = (arguments.get("content") or "").strip()
                if not content:
                    return jsonrpc_error(_id, -32602, "content required")
                data = await pushplus_notify(title=title, content=content)
                return jsonrpc_result(_id, {"content": [{"type":"text","text": json.dumps(data, ensure_ascii=False)}]})
            if name == "schedule_pushplus":
                title = (arguments.get("title") or "").strip()
                content = (arguments.get("content") or "").strip()
                run_at = (arguments.get("run_at") or "").strip()
                repeat = (arguments.get("repeat") or "none").strip().lower()
                template = (arguments.get("template") or "txt").strip().lower()

                if not title or not content or not run_at:
                    return jsonrpc_error(_id, -32602, "title/content/run_at required")

                job = await create_push_schedule(title=title, content=content, run_at=run_at, repeat=repeat, template=template)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(job, ensure_ascii=False)}]})

            if name == "list_pushplus_schedules":
                limit = int(arguments.get("limit") or 20)
                jobs = await list_push_schedules(limit=limit)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(jobs, ensure_ascii=False)}]})

            if name == "cancel_pushplus_schedule":
                job_id = (arguments.get("id") or "").strip()
                if not job_id:
                    return jsonrpc_error(_id, -32602, "id required")
                out = await cancel_push_schedule(job_id)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})

            if name == "run_due_pushplus":
                out = await run_due_push_schedules()
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
            return jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            return jsonrpc_error(_id, -32000, f"Tool execution error: {e}", data=_tb.format_exc())

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
"",
"-- PushPlus schedules (for /cron/tick)",
"create table if not exists push_schedules (",
"  id uuid primary key default gen_random_uuid(),",
"  title text not null,",
"  content text not null,",
"  template text not null default 'txt',",
"  run_at timestamptz not null,",
"  repeat text not null default 'none',",
"  enabled boolean not null default true,",
"  last_run_at timestamptz,",
"  created_at timestamptz not null default now()",
");",
"create index if not exists push_schedules_run_at_idx on push_schedules (run_at) where enabled = true;",
        ],
    }



# -----------------------
# AMap (Gaode) helpers
# -----------------------
AMAP_BASE = "https://restapi.amap.com"

async def _amap_get(path: str, params: dict):
    if not AMAP_KEY:
        raise RuntimeError("AMAP_KEY missing")
    url = AMAP_BASE + path
    p = {"key": AMAP_KEY, **{k:v for k,v in (params or {}).items() if v is not None and v != ""}}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=p)
        r.raise_for_status()
        return r.json()

async def amap_geocode(address: str, city: Optional[str] = None):
    # https://restapi.amap.com/v3/geocode/geo
    return await _amap_get("/v3/geocode/geo", {"address": address, "city": city})

async def amap_reverse_geocode(location: str):
    # https://restapi.amap.com/v3/geocode/regeo
    return await _amap_get("/v3/geocode/regeo", {"location": location, "radius": 1000, "extensions": "all"})

async def amap_weather(city: str, extensions: str = "base"):
    # https://restapi.amap.com/v3/weather/weatherInfo
    if extensions not in ("base", "all"):
        extensions = "base"
    return await _amap_get("/v3/weather/weatherInfo", {"city": city, "extensions": extensions})

async def amap_poi_around(location: str, keywords: Optional[str] = None, types: Optional[str] = None, radius: int = 3000):
    # https://restapi.amap.com/v3/place/around
    return await _amap_get("/v3/place/around", {
        "location": location,
        "keywords": keywords,
        "types": types,
        "radius": radius,
        "sortrule": "distance",
        "offset": 10,
        "page": 1
    })

async def amap_route_driving(origin: str, destination: str):
    # https://restapi.amap.com/v3/direction/driving
    return await _amap_get("/v3/direction/driving", {
        "origin": origin,
        "destination": destination,
        "strategy": 0,
        "extensions": "base"
    })

# -----------------------
# PushPlus helper
# -----------------------
async def pushplus_notify(title: str, content: str, template: str = "txt"):
    if not PUSHPLUS_TOKEN:
        raise RuntimeError("PUSHPLUS_TOKEN missing")

    url = "https://www.pushplus.plus/send"

    payload = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": template,
        "channel": "app"
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()
# ----------------------------
# PushPlus scheduler (simple)
# ----------------------------

PUSH_SCHEDULE_TABLE = os.getenv("PUSH_SCHEDULE_TABLE", "push_schedules")
CRON_SECRET = os.getenv("CRON_SECRET", "").strip()

def _parse_run_at(run_at: str) -> str:
    """
    Accepts:
      - 2026-02-25 08:30:00+08:00
      - 2026-02-25T08:30:00+08:00
      - 2026-02-25 08:30:00  (treated as Asia/Shanghai +08:00)
      - 2026-02-25T08:30:00 (treated as Asia/Shanghai +08:00)
    Returns ISO string with offset.
    """
    s = run_at.strip()
    s = s.replace(" ", "T", 1) if " " in s and "T" not in s else s
    if s.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", s):
        return s
    # naive -> assume +08:00
    return s + "+08:00"

def _next_run_iso(prev_run_iso: str, repeat: str) -> str | None:
    try:
        # python 3.11: fromisoformat accepts "+08:00"
        dt = datetime.fromisoformat(prev_run_iso.replace("Z", "+00:00"))
    except Exception:
        return None

    repeat = (repeat or "none").lower()
    if repeat == "hourly":
        dt = dt + timedelta(hours=1)
    elif repeat == "daily":
        dt = dt + timedelta(days=1)
    elif repeat == "weekly":
        dt = dt + timedelta(days=7)
    else:
        return None
    return dt.isoformat()

async def create_push_schedule(*, title: str, content: str, run_at: str, repeat: str = "none", template: str = "txt") -> dict:
    run_at_iso = _parse_run_at(run_at)
    payload = {
        "title": title,
        "content": content,
        "run_at": run_at_iso,
        "repeat": (repeat or "none").lower(),
        "template": (template or "txt").lower(),
        "enabled": True,
    }
    url = f"{SUPABASE_URL}/rest/v1/{PUSH_SCHEDULE_TABLE}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers={**supabase_headers(), "Prefer": "return=representation"}, json=payload)
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else payload

async def list_push_schedules(*, limit: int = 20) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/{PUSH_SCHEDULE_TABLE}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            url,
            headers=supabase_headers(),
            params={"select": "id,title,run_at,repeat,enabled,created_at,last_run_at", "order": "created_at.desc", "limit": str(limit)},
        )
        r.raise_for_status()
        return r.json()

async def cancel_push_schedule(job_id: str) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{PUSH_SCHEDULE_TABLE}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.patch(
            url,
            headers={**supabase_headers(), "Prefer": "return=representation"},
            params={"id": f"eq.{job_id}"},
            json={"enabled": False},
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else {"id": job_id, "enabled": False}

async def run_due_push_schedules() -> dict:
    """
    Pull due jobs from Supabase and push via PushPlus.
    Returns a dict for debugging (never raises to FastAPI).
    """
    # Always compare in UTC because Supabase `timestamptz` is stored in UTC.
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    now_iso = now_utc.isoformat().replace("+00:00", "Z")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) Fetch due jobs
        try:
            url = f"{SUPABASE_URL}/rest/v1/push_schedules"
            params = {
                "select": "*",
                "enabled": "eq.true",
                "run_at": f"lte.{now_iso}",
                # Only pick pending jobs (or rows without status yet)
                "or": "(status.eq.pending,status.is.null)",
                "order": "run_at.asc",
                "limit": "50",
            }
            r = await client.get(url, headers=supabase_headers(), params=params, timeout=20.0)
        except Exception as e:
            return {"ok": False, "now": now_iso, "stage": "fetch_due_jobs", "error": f"{type(e).__name__}: {e}"}

        if r.status_code >= 400:
            return {"ok": False, "now": now_iso, "stage": "fetch_due_jobs", "http_status": r.status_code, "body": r.text[:2000]}

        try:
            jobs = r.json() or []
        except Exception as e:
            return {"ok": False, "now": now_iso, "stage": "parse_due_jobs", "http_status": r.status_code, "body": r.text[:2000], "error": f"{type(e).__name__}: {e}"}

        sent = 0
        touched = 0
        errors = []

        # 2) Process jobs in order
        for job in jobs:
            job_id = job.get("id")
            title = job.get("title") or "提醒"
            content = job.get("content") or ""
            template = job.get("template") or "html"
            repeat = (job.get("repeat") or "none").lower()

            if not job_id:
                continue

            # Mark as running to reduce duplicate sends in rare concurrent ticks
            try:
                patch_url = f"{SUPABASE_URL}/rest/v1/push_schedules?id=eq.{job_id}"
                await client.patch(
                    patch_url,
                    headers={**supabase_headers(), "Prefer": "return=minimal"},
                    json={"status": "running"},
                    timeout=20.0,
                )
            except Exception:
                # Not fatal; continue
                pass

            # 2.1 Send push
            try:
                await pushplus_notify(title=title, content=content, template=template)
                sent += 1
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                errors.append({"id": job_id, "stage": "pushplus", "error": err_msg})
                # Save error back to DB
                try:
                    patch_url = f"{SUPABASE_URL}/rest/v1/push_schedules?id=eq.{job_id}"
                    await client.patch(
                        patch_url,
                        headers={**supabase_headers(), "Prefer": "return=minimal"},
                        json={"status": "error", "last_error": err_msg[:1800]},
                        timeout=20.0,
                    )
                    touched += 1
                except Exception:
                    pass
                continue

            # 2.2 Update schedule (IMPORTANT: next run is based on the scheduled run_at, not 'now')
            try:
                patch: Dict[str, Any] = {
                    "last_run_at": now_iso,
                    "sent_at": now_iso,
                    "last_error": None,
                }

                if repeat and repeat != "none":
                    # Keep the same clock time by stepping from the *previous run_at*
                    prev_run_at = job.get("run_at") or now_iso
                    patch["run_at"] = _next_run_iso(prev_run_at, repeat)
                    patch["status"] = "pending"
                    patch["enabled"] = True
                else:
                    patch["enabled"] = False
                    patch["status"] = "sent"

                patch_url = f"{SUPABASE_URL}/rest/v1/push_schedules?id=eq.{job_id}"
                pr = await client.patch(
                    patch_url,
                    headers={**supabase_headers(), "Prefer": "return=minimal"},
                    json=patch,
                    timeout=20.0,
                )
                if pr.status_code >= 400:
                    errors.append({"id": job_id, "stage": "update_schedule", "http_status": pr.status_code, "body": pr.text[:1000]})
                else:
                    touched += 1
            except Exception as e:
                errors.append({"id": job_id, "stage": "update_schedule", "error": f"{type(e).__name__}: {e}"})

        return {"ok": True, "now": now_iso, "checked": len(jobs), "sent": sent, "touched": touched, "errors": errors}
@app.get("/cron/tick")
async def cron_tick(secret: str = ""):
    # 用外部定时器（cron-job.org / UptimeRobot / GitHub Actions）每分钟打这个接口
    if CRON_SECRET and secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")
    return await run_due_push_schedules()
