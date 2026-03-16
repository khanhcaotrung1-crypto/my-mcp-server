import os
import json
import asyncio
import uuid
import re
from datetime import datetime, timezone, timedelta

SH_TZ = timezone(timedelta(hours=8))  # Asia/Shanghai fixed offset
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
NOTION_TOKEN = (os.getenv("NOTION_TOKEN") or "").strip()
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()
SILICONFLOW_KEY = (os.getenv("SILICONFLOW_KEY") or "").strip()

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
    "name": "forget_memory",
    "description": "手动把一条记忆标记为「淡忘」，之后不会再被召回",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "记忆的 UUID"}
        },
        "required": ["memory_id"]
    }
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
{
    "name": "notion_create_page",
    "description": "在 Notion 指定页面下创建一个新子页面（适合写日记、新建笔记）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "parent_page_id": {"type": "string", "description": "父页面的 Notion page ID（32位字符串，从页面URL中获取）"},
            "title": {"type": "string", "description": "新页面标题"},
            "content": {"type": "string", "description": "页面正文内容（纯文本，支持换行）"}
        },
        "required": ["parent_page_id", "title"]
    }
},
{
    "name": "notion_append_content",
    "description": "在 Notion 已有页面末尾追加文字内容（适合在日记/笔记里续写）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "page_id": {"type": "string", "description": "目标页面的 Notion page ID"},
            "content": {"type": "string", "description": "要追加的文字内容（纯文本，支持换行）"}
        },
        "required": ["page_id", "content"]
    }
},
{
    "name": "notion_search",
    "description": "在 Notion 工作空间中搜索页面（按标题关键词）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "limit": {"type": "integer", "description": "最多返回几条，默认10", "default": 10}
        },
        "required": ["query"]
    }
},
{
    "name": "notion_read_page",
    "description": "读取 Notion 页面的文字内容（需要先用 notion_search 找到 page_id）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "page_id": {"type": "string", "description": "页面的 Notion page ID"},
            "max_chars": {"type": "integer", "description": "最多返回多少字符，默认3000", "default": 3000}
        },
        "required": ["page_id"]
    }
},
{
    "name": "web_search",
    "description": "在互联网上搜索信息（使用 Tavily，适合查新闻、查资料、查事实）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或问题"},
            "max_results": {"type": "integer", "description": "最多返回几条结果，默认5", "default": 5}
        },
        "required": ["query"]
    }
},
{
    "name": "add_note",
    "description": "添加一条便签或待办事项",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "便签内容"}
        },
        "required": ["content"]
    }
},
{
    "name": "list_notes",
    "description": "查看当前未完成的便签/待办列表",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
},
{
    "name": "done_note",
    "description": "把一条便签标记为已完成",
    "inputSchema": {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "便签的 UUID"}
        },
        "required": ["note_id"]
    }
},
{
    "name": "update_core_block",
    "description": "更新 Core Blocks 核心档案（AI 自动分析对话后调用，更新某个核心信息块）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "block_key": {"type": "string", "description": "要更新的块名，如 relationship / user_profile / rituals"},
            "content": {"type": "string", "description": "新的内容"}
        },
        "required": ["block_key", "content"]
    }
},
{
    "name": "get_core_blocks",
    "description": "读取当前所有 Core Blocks 核心档案内容",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
},
{
    "name": "append_diary",
    "description": "写一篇日记到数据库，记录当天发生的事、感受或想法。每次对话结束时如有值得记录的内容可主动调用。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "日记正文内容"},
            "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD，留空则用今天"}
        },
        "required": ["content"]
    }
},
{
    "name": "list_diary",
    "description": "查询历史日记，可按日期范围筛选",
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "返回条数，默认10"},
            "date_from": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
            "date_to": {"type": "string", "description": "结束日期 YYYY-MM-DD"}
        }
    }
},
{
    "name": "log_mood",
    "description": "记录当下情绪状态。当用户表达明显情绪时可主动调用。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "mood": {"type": "string", "description": "情绪标签，如 happy/sad/anxious/calm/excited/tired 等"},
            "intensity": {"type": "integer", "description": "强度 1-5，5最强"},
            "note": {"type": "string", "description": "补充说明，可留空"}
        },
        "required": ["mood", "intensity"]
    }
},
{
    "name": "get_mood_history",
    "description": "查询情绪记录历史",
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "返回条数，默认20"},
            "mood": {"type": "string", "description": "按情绪标签筛选，留空返回全部"}
        }
    }
},
{
    "name": "search_note",
    "description": "按关键词搜索 notes 待办/备忘内容",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "include_done": {"type": "boolean", "description": "是否包含已完成的，默认 false"}
        },
        "required": ["query"]
    }
},
{
    "name": "get_memory_stats",
    "description": "查看记忆库统计：总数、权重分布、各分类数量、濒临遗忘的记忆",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
},
{
    "name": "get_phone_status",
    "description": "查询念念手机的实时状态：当前位置、电量、正在用的APP、是否充电、WiFi等。当对话中涉及念念在哪、在干什么、状态如何时主动调用。",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
},
{
    "name": "get_weather",
    "description": "直接用城市名查询天气，封装了地理编码步骤",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如 北京、上海、成都"},
            "forecast": {"type": "boolean", "description": "true 返回未来几天预报，false 返回实时天气，默认 false"}
        },
        "required": ["city"]
    }
},
{
    "name": "generate_card",
    "description": "生成一张精美的 HTML 卡片，可用于纪念日、情书/表白、自定义场景。返回 HTML 字符串，RikkaHub 会渲染展示。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "card_type": {
                "type": "string",
                "description": "卡片类型：anniversary（纪念日）/ love_letter（情书/表白）/ custom（自定义）",
                "enum": ["anniversary", "love_letter", "custom"]
            },
            "title": {"type": "string", "description": "卡片标题"},
            "body": {"type": "string", "description": "卡片正文内容"},
            "footer": {"type": "string", "description": "底部落款或日期，可留空"},
            "accent": {"type": "string", "description": "custom 类型时可指定主题色，如 #f9a8d4，留空自动"}
        },
        "required": ["card_type", "title", "body"]
    }
},
{
    "name": "generate_image",
    "description": "根据描述词生成一张图片，返回图片URL。当用户要求画图、生成图片、或你自己想配图时调用。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "图片描述词，建议用英文以获得更好效果"},
            "size": {"type": "string", "description": "图片尺寸，可选：1024x1024 / 768x1344 / 1344x768，默认1024x1024", "default": "1024x1024"}
        },
        "required": ["prompt"]
    }
}
]

# ---------- SSE sessions ----------
SESSIONS: Dict[str, "asyncio.Queue[dict]"] = {}


async def _save_session(session_id: str):
    """把 session_id 持久化到 Supabase"""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/mcp_sessions",
                headers={**_supabase_headers(), "Prefer": "return=minimal"},
                json={"session_id": session_id}
            )
    except Exception as e:
        print(f"[MCP] save_session error: {e}", flush=True)


async def _session_exists_in_db(session_id: str) -> bool:
    """检查 session_id 是否在 Supabase 里"""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/mcp_sessions",
                headers=_supabase_headers(),
                params={"session_id": f"eq.{session_id}", "select": "session_id", "limit": "1"}
            )
        return r.status_code == 200 and bool(r.json())
    except Exception:
        return False

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
        # 通知消息无需响应，返回空 204
        if resp is None:
            from fastapi.responses import Response
            return Response(status_code=204)
        return JSONResponse(resp, status_code=200)

    # 2) 否则当作“握手”请求：创建 session，返回 endpoint 信息（非流式）
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = asyncio.Queue()
    await _save_session(session_id)
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

@app.get("/debug/notion_search")
async def debug_notion_search(q: str = "姐姐"):
    try:
        url = f"{NOTION_BASE}/search"
        body = {
            "query": q,
            "filter": {"value": "page", "property": "object"},
            "page_size": 5,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=_notion_headers(), json=body)
        return {"status": r.status_code, "raw": r.json()}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

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


# mcp_sse 是内部 helper，由 mcp_entry 的 GET 分支调用，不单独注册路由
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
# ============================================================
# 手机状态感知
# ============================================================

@app.post("/device/status")
async def receive_device_status(request: Request):
    """MacroDroid 上报手机状态的接口"""
    try:
        data = await request.json()
    except Exception:
        data = dict(request.query_params)

    battery   = data.get("battery", "")
    charging  = data.get("charging", "")
    wifi      = data.get("wifi", "")
    app       = data.get("app", "")
    activity  = data.get("activity", "")   # 移动速度 km/h
    screen    = data.get("screen", "")
    address   = data.get("address", "")    # 经纬度字符串 lat,lng

    # 逆地理编码（有高德key才做）
    location_text = ""
    AMAP_KEY = os.environ.get("AMAP_KEY", "")
    if address and AMAP_KEY:
        try:
            parts = address.replace(" ", "").split(",")
            if len(parts) == 2:
                lat, lng = parts[0], parts[1]
                # 高德需要 lng,lat
                lng_lat = f"{lng},{lat}"
                async with httpx.AsyncClient(timeout=8) as client:
                    r = await client.get(
                        "https://restapi.amap.com/v3/geocode/regeo",
                        params={"key": AMAP_KEY, "location": lng_lat, "extensions": "base"}
                    )
                if r.status_code == 200:
                    rj = r.json()
                    if rj.get("status") == "1":
                        location_text = rj["regeocode"].get("formatted_address", "")
        except Exception as e:
            print(f"[device] 逆地理编码失败: {e}", flush=True)

    # 组装状态文本
    parts = []
    if location_text:
        parts.append(f"位置：{location_text}")
    elif address:
        parts.append(f"坐标：{address}")
    if battery:
        charge_str = "充电中" if str(charging).lower() in ("true", "1", "yes") else "未充电"
        parts.append(f"电量：{battery}%（{charge_str}）")
    if wifi:
        parts.append(f"WiFi：{wifi}")
    if screen:
        parts.append(f"屏幕：{screen}")
    if app:
        parts.append(f"正在使用：{app}")
    if activity:
        parts.append(f"移动速度：{activity} km/h")

    status_text = "、".join(parts) if parts else "状态未知"

    # 存到 Supabase phone_status 表
    url = f"{SUPABASE_URL}/rest/v1/phone_status"
    payload = {
        "status_text": status_text,
        "battery": int(battery) if str(battery).isdigit() else None,
        "charging": str(charging).lower() in ("true", "1", "yes"),
        "wifi": wifi or None,
        "app": app or None,
        "screen": screen or None,
        "location": location_text or address or None,
        "raw": json.dumps(data, ensure_ascii=False)
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            url,
            headers={**_supabase_headers(), "Prefer": "return=minimal"},
            json=payload
        )

    print(f"[device] 状态上报: {status_text} | Supabase {r.status_code}", flush=True)
    return {"status": "ok", "parsed": status_text}


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
        # 内存里没有，去 Supabase 查是否是重启前的合法 session
        exists = await _session_exists_in_db(session_id)
        if not exists:
            raise HTTPException(status_code=404, detail="Unknown session")
        # 合法 session，重建队列
        SESSIONS[session_id] = asyncio.Queue()
        print(f"[MCP] session {session_id[:8]}... restored from DB", flush=True)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    resp = await handle_rpc(payload)

    # 通知消息不需要回复，直接返回 200
    if resp is None:
        return JSONResponse({"ok": True}, status_code=200)

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

    last_err = None
    for attempt in range(3):  # 最多重试3次
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"Embeddings HTTP {r.status_code}: {r.text}")
            data = r.json()
            try:
                return data["data"][0]["embedding"]
            except Exception:
                raise RuntimeError(f"Bad embeddings response: {data}")
        except httpx.TimeoutException as e:
            last_err = e
            print(f"[embed] timeout on attempt {attempt + 1}, retrying...", flush=True)
            await asyncio.sleep(1)
        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            print(f"[embed] error on attempt {attempt + 1}: {e}, retrying...", flush=True)
            await asyncio.sleep(1)

    raise RuntimeError(f"embed_text failed after 3 attempts: {last_err}")


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
    imp = importance if importance is not None else 3
    initial_weight = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.3}.get(int(imp), 0.6)
    payload: Dict[str, Any] = {
        "title": title,
        "content": content,
        "weight": initial_weight,
        "recall_count": 0,
        "forgotten": False,
    }
    if category is not None:
        payload["category"] = category
    if importance is not None:
        payload["importance"] = importance
    if embedding_vec is not None:
        # PostgREST REST insert 需要传 JSON 数组，不能用 pgvector 字符串格式
        payload["embedding"] = embedding_vec

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
    # 转义 PostgREST 特殊字符，防止过滤表达式被破坏
    safe_query = query.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace(",", "\\,").replace("*", "\\*")
    # title/content ILIKE
    params = {
        "select": "id,title,content,category,importance,created_at",
        "or": f"(title.ilike.*{safe_query}*,content.ilike.*{safe_query}*)",
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
        # PostgREST RPC 调用 vector 参数需要传 JSON 数组，不能用 pgvector 字符串格式
        "query_embedding": query_embedding,
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

    # MCP 协议：通知消息（notifications/*）id 为 null，按规范不能返回任何响应
    if method is None or (isinstance(method, str) and method.startswith("notifications/")):
        return None

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

            if name == "forget_memory":
                memory_id = (arguments.get("memory_id") or "").strip()
                if not memory_id:
                    return jsonrpc_error(_id, -32602, "memory_id required")
                url = f"{SUPABASE_URL}/rest/v1/memories?id=eq.{memory_id}"
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.patch(url, headers=_supabase_headers(), json={"forgotten": True, "weight": 0.0})
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": "记忆已淡忘 ✓"}]})

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
                    rows = await supabase_keyword_search(query, k=k)

                # 异步更新被召回记忆的权重（fire and forget）
                async def _boost_recalled(recalled_rows):
                    try:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        for row in recalled_rows:
                            rid = row.get("id")
                            if not rid:
                                continue
                            cur_weight = float(row.get("weight") or 0.6)
                            cur_count = int(row.get("recall_count") or 0)
                            new_weight = min(1.0, cur_weight + 0.05)
                            patch_url = f"{SUPABASE_URL}/rest/v1/memories?id=eq.{rid}"
                            async with httpx.AsyncClient(timeout=5) as _c:
                                await _c.patch(patch_url, headers=_supabase_headers(), json={
                                    "weight": new_weight,
                                    "recall_count": cur_count + 1,
                                    "last_recalled_at": now_iso,
                                })
                    except Exception:
                        pass
                asyncio.create_task(_boost_recalled(rows))

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
                page = int(arguments.get("page") or 1)
                offset = int(arguments.get("offset") or 10)
                data = await amap_poi_around(location=location, keywords=keywords, types=types, radius=radius, page=page, offset=offset)
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
                    dlng = arguments.get("dest_lng")
                    dlat = arguments.get("dest_lat")
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

            # -----------------
            # Notion tools
            # -----------------
            if name == "notion_create_page":
                parent_page_id = (arguments.get("parent_page_id") or "").strip().replace("-", "")
                title = (arguments.get("title") or "").strip()
                content = (arguments.get("content") or "").strip()
                if not parent_page_id or not title:
                    return jsonrpc_error(_id, -32602, "parent_page_id and title required")
                data = await notion_create_page(parent_page_id=parent_page_id, title=title, content=content)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "notion_append_content":
                page_id = (arguments.get("page_id") or "").strip().replace("-", "")
                content = (arguments.get("content") or "").strip()
                if not page_id or not content:
                    return jsonrpc_error(_id, -32602, "page_id and content required")
                data = await notion_append_content(page_id=page_id, content=content)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "notion_read_page":
                page_id = (arguments.get("page_id") or "").strip().replace("-", "")
                if not page_id:
                    return jsonrpc_error(_id, -32602, "page_id required")
                max_chars = int(arguments.get("max_chars") or 3000)
                data = await notion_read_page(page_id=page_id, max_chars=max_chars)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "notion_search":
                query = (arguments.get("query") or "").strip()
                if not query:
                    return jsonrpc_error(_id, -32602, "query required")
                limit = int(arguments.get("limit") or 10)
                data = await notion_search(query=query, limit=limit)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "web_search":
                query = (arguments.get("query") or "").strip()
                if not query:
                    return jsonrpc_error(_id, -32602, "query required")
                max_results = int(arguments.get("max_results") or 5)
                data = await tavily_search(query=query, max_results=max_results)
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]})

            if name == "add_note":
                content = (arguments.get("content") or "").strip()
                if not content:
                    return jsonrpc_error(_id, -32602, "content required")
                url = f"{SUPABASE_URL}/rest/v1/notes"
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(url, headers=_supabase_headers(), json={"content": content})
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": f"便签已添加：{content}"}]})

            if name == "list_notes":
                url = f"{SUPABASE_URL}/rest/v1/notes"
                params = {"select": "id,content,created_at", "done": "eq.false", "order": "created_at.desc", "limit": "20"}
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(url, headers=_supabase_headers(), params=params)
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(r.json(), ensure_ascii=False)}]})

            if name == "done_note":
                note_id = (arguments.get("note_id") or "").strip()
                if not note_id:
                    return jsonrpc_error(_id, -32602, "note_id required")
                url = f"{SUPABASE_URL}/rest/v1/notes?id=eq.{note_id}"
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.patch(url, headers=_supabase_headers(), json={"done": True, "done_at": datetime.now(timezone.utc).isoformat()})
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": "便签已完成 ✓"}]})

            if name == "update_core_block":
                block_key = (arguments.get("block_key") or "").strip()
                content = (arguments.get("content") or "").strip()
                if not block_key or not content:
                    return jsonrpc_error(_id, -32602, "block_key and content required")
                url = f"{SUPABASE_URL}/rest/v1/core_blocks?block_key=eq.{block_key}"
                patch_headers = {**_supabase_headers(), "Prefer": "return=representation"}
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.patch(url, headers=patch_headers, json={"content": content, "updated_at": datetime.now(timezone.utc).isoformat()})
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                if r.json() == []:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r2 = await client.post(f"{SUPABASE_URL}/rest/v1/core_blocks", headers=_supabase_headers(), json={"block_key": block_key, "content": content})
                    if r2.status_code >= 400:
                        raise RuntimeError(f"Supabase insert {r2.status_code}: {r2.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": f"Core Block [{block_key}] 已更新"}]})

            if name == "get_core_blocks":
                url = f"{SUPABASE_URL}/rest/v1/core_blocks"
                params = {"select": "block_key,content,updated_at", "order": "priority.asc"}
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(url, headers=_supabase_headers(), params=params)
                if r.status_code >= 400:
                    raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(r.json(), ensure_ascii=False)}]})

            if name == "append_diary":
                result = await tool_append_diary(
                    content=arguments.get("content", ""),
                    date=arguments.get("date")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": result}]})

            if name == "list_diary":
                result = await tool_list_diary(
                    limit=arguments.get("limit", 10),
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "log_mood":
                result = await tool_log_mood(
                    mood=arguments.get("mood", ""),
                    intensity=arguments.get("intensity", 3),
                    note=arguments.get("note")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": result}]})

            if name == "get_mood_history":
                result = await tool_get_mood_history(
                    limit=arguments.get("limit", 20),
                    mood=arguments.get("mood")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "search_note":
                result = await tool_search_note(
                    query=arguments.get("query", ""),
                    include_done=arguments.get("include_done", False)
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "get_memory_stats":
                result = await tool_get_memory_stats()
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "get_phone_status":
                result = await tool_get_phone_status()
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "get_weather":
                result = await tool_get_weather(
                    city=arguments.get("city", ""),
                    forecast=arguments.get("forecast", False)
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

            if name == "generate_card":
                result = tool_generate_card(
                    card_type=arguments.get("card_type", "custom"),
                    title=arguments.get("title", ""),
                    body=arguments.get("body", ""),
                    footer=arguments.get("footer", ""),
                    accent=arguments.get("accent", "")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": result}]})

            if name == "generate_image":
                result = await tool_generate_image(
                    prompt=arguments.get("prompt", ""),
                    size=arguments.get("size", "1024x1024")
                )
                return jsonrpc_result(_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})

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
# Notion helpers
# -----------------------
NOTION_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

def _notion_headers():
    if not NOTION_TOKEN:
        raise RuntimeError("NOTION_TOKEN missing")
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

def _text_to_notion_blocks(text: str) -> list:
    """把纯文本按换行拆成 paragraph blocks"""
    blocks = []
    for line in text.split("\n"):
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": line}}] if line else []
            }
        })
    return blocks

async def notion_create_page(parent_page_id: str, title: str, content: str = "") -> dict:
    url = f"{NOTION_BASE}/pages"
    body: dict = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}]
            }
        },
    }
    if content:
        body["children"] = _text_to_notion_blocks(content)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_notion_headers(), json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Notion API {r.status_code}: {r.text}")
    data = r.json()
    return {"id": data.get("id"), "url": data.get("url"), "title": title}

async def notion_append_content(page_id: str, content: str) -> dict:
    url = f"{NOTION_BASE}/blocks/{page_id}/children"
    body = {"children": _text_to_notion_blocks(content)}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.patch(url, headers=_notion_headers(), json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Notion API {r.status_code}: {r.text}")
    return {"ok": True, "appended_blocks": len(body["children"])}

async def notion_search(query: str, limit: int = 10) -> list:
    url = f"{NOTION_BASE}/search"
    body = {
        "query": query,
        "filter": {"value": "page", "property": "object"},
        "page_size": max(1, min(limit, 20)),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_notion_headers(), json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Notion API {r.status_code}: {r.text}")
    results = r.json().get("results", [])
    simplified = []
    for page in results:
        props = page.get("properties", {})
        title_field = None
        for field in props.values():
            if field.get("type") == "title":
                title_field = field
                break
        title_list = title_field.get("title", []) if title_field else []
        title = title_list[0]["plain_text"] if title_list else "(无标题)"
        simplified.append({
            "id": page.get("id", "").replace("-", ""),
            "title": title,
            "url": page.get("url", ""),
            "last_edited": page.get("last_edited_time", ""),
        })
    return simplified
    

async def notion_read_page(page_id: str, max_chars: int = 3000) -> dict:
    url = f"{NOTION_BASE}/blocks/{page_id}/children"
    all_text: list = []
    next_cursor = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            req_params: dict = {"page_size": 100}
            if next_cursor:
                req_params["start_cursor"] = next_cursor
            r = await client.get(url, headers=_notion_headers(), params=req_params)
            if r.status_code >= 400:
                raise RuntimeError(f"Notion API {r.status_code}: {r.text}")
            data = r.json()
            for block in data.get("results", []):
                btype = block.get("type", "")
                rich_text = block.get(btype, {}).get("rich_text", [])
                line = "".join(t.get("plain_text", "") for t in rich_text)
                if line:
                    all_text.append(line)
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")
    full_text = "\n".join(all_text)
    return {
        "page_id": page_id,
        "content": full_text[:max_chars],
        "truncated": len(full_text) > max_chars,
        "total_chars": len(full_text),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_notion_headers(), json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Notion API {r.status_code}: {r.text}")
    results = r.json().get("results", [])
    simplified = []
    for page in results:
        title_list = (
            page.get("properties", {}).get("title", {}).get("title", [])
            or page.get("properties", {}).get("名称", {}).get("title", [])
        )
        title = title_list[0]["plain_text"] if title_list else "(无标题)"
        simplified.append({
            "id": page.get("id", "").replace("-", ""),
            "title": title,
            "url": page.get("url", ""),
            "last_edited": page.get("last_edited_time", ""),
        })
    return simplified


# -----------------------
# Tavily (web search) helpers
# -----------------------
async def tavily_search(query: str, max_results: int = 5) -> dict:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY missing")
    url = "https://api.tavily.com/search"
    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max(1, min(max_results, 10)),
        "search_depth": "basic",
        "include_answer": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Tavily API {r.status_code}: {r.text}")
    data = r.json()
    return {
        "answer": data.get("answer", ""),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
            }
            for item in data.get("results", [])
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

async def amap_reverse_geocode(location: str, radius: int = 1000, extensions: str = "all"):
    # https://restapi.amap.com/v3/geocode/regeo
    if extensions not in ("base", "all"):
        extensions = "all"
    return await _amap_get("/v3/geocode/regeo", {"location": location, "radius": radius, "extensions": extensions})

async def amap_weather(city: str, extensions: str = "base"):
    # https://restapi.amap.com/v3/weather/weatherInfo
    if extensions not in ("base", "all"):
        extensions = "base"
    return await _amap_get("/v3/weather/weatherInfo", {"city": city, "extensions": extensions})

async def amap_poi_around(location: str, keywords: Optional[str] = None, types: Optional[str] = None, radius: int = 3000, page: int = 1, offset: int = 10):
    # https://restapi.amap.com/v3/place/around
    return await _amap_get("/v3/place/around", {
        "location": location,
        "keywords": keywords,
        "types": types,
        "radius": radius,
        "sortrule": "distance",
        "offset": offset,
        "page": page
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

def _next_run_iso(prev_run_iso: str, repeat: str, now_iso_utc: str) -> Optional[str]:
    """Return the *next* run_at strictly after `now_iso_utc` (UTC), keeping the original local time-of-day.

    Why: if a daily job is months behind, doing +1 day will spam (catch-up loop). We instead jump to the next occurrence.
    """
    try:
        prev_dt_utc = datetime.fromisoformat(prev_run_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        now_dt_utc = datetime.fromisoformat(now_iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

    repeat = (repeat or "none").lower()
    # derive desired local time-of-day (Asia/Shanghai) from previous run_at
    prev_local = prev_dt_utc.astimezone(SH_TZ)
    now_local = now_dt_utc.astimezone(SH_TZ)

    if repeat == "hourly":
        cand = now_local.replace(minute=prev_local.minute, second=prev_local.second, microsecond=0)
        if cand <= now_local:
            cand = cand + timedelta(hours=1)

    elif repeat == "daily":
        cand = now_local.replace(hour=prev_local.hour, minute=prev_local.minute, second=prev_local.second, microsecond=0)
        if cand <= now_local:
            cand = cand + timedelta(days=1)

    elif repeat == "weekly":
        # next same weekday + time
        cand = now_local.replace(hour=prev_local.hour, minute=prev_local.minute, second=prev_local.second, microsecond=0)
        target_wd = prev_local.weekday()  # Monday=0
        days_ahead = (target_wd - cand.weekday()) % 7
        if days_ahead == 0 and cand <= now_local:
            days_ahead = 7
        cand = cand + timedelta(days=days_ahead)

    else:
        return None

    return cand.astimezone(timezone.utc).isoformat()

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
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
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
                    next_run = _next_run_iso(prev_run_at, repeat, now_iso)
                    if next_run is not None:
                        patch["run_at"] = next_run
                        patch["status"] = "pending"
                        patch["enabled"] = True
                    else:
                        # 解析失败则禁用，防止任务变成 null run_at 僵尸
                        patch["enabled"] = False
                        patch["status"] = "error"
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


async def decay_memory_weights() -> dict:
    """每天衰减一次记忆权重，低于 0.05 的标记为淡忘"""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return {"skipped": True, "reason": "supabase not configured"}

    DECAY_RATE = 0.02        # 每天降低 0.02
    FORGET_THRESHOLD = 0.05  # 低于此值标记为淡忘

    url = f"{SUPABASE_URL}/rest/v1/memories"
    params = {
        "select": "id,weight",
        "forgotten": "eq.false",
        "weight": "gt.0",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        return {"error": f"fetch failed {r.status_code}"}

    rows = r.json()
    decayed = 0
    forgotten = 0

    for row in rows:
        rid = row.get("id")
        cur_weight = float(row.get("weight") or 0.6)
        new_weight = round(max(0.0, cur_weight - DECAY_RATE), 4)
        patch = {"weight": new_weight}
        if new_weight < FORGET_THRESHOLD:
            patch["forgotten"] = True
            forgotten += 1
        else:
            decayed += 1
        patch_url = f"{SUPABASE_URL}/rest/v1/memories?id=eq.{rid}"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.patch(patch_url, headers=_supabase_headers(), json=patch)

    return {"decayed": decayed, "forgotten": forgotten, "total": len(rows)}


@app.get("/cron/tick")
async def cron_tick(request: Request):
    # 用外部定时器（cron-job.org / UptimeRobot / GitHub Actions）每分钟打这个接口
    # secret 通过 Authorization header 传递，避免出现在 URL 日志中
    # 调用方式：curl -H "Authorization: Bearer <CRON_SECRET>" https://your-server/cron/tick
    if CRON_SECRET:
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if token != CRON_SECRET:
            raise HTTPException(status_code=401, detail="bad secret")

    push_result = await run_due_push_schedules()

    # 每天只跑一次权重衰减（整点 00 分触发）
    now_sh = datetime.now(SH_TZ)
    decay_result = {}
    cleanup_result = {}
    if now_sh.hour == 3 and now_sh.minute == 0:
        decay_result = await decay_memory_weights()
        cleanup_result = await cleanup_old_mood_logs(days=90)

    return {**push_result, "decay": decay_result, "cleanup": cleanup_result}


# ============================================================
# 新增工具实现
# ============================================================

async def tool_append_diary(content: str, date: Optional[str] = None) -> str:
    """写日记"""
    from datetime import date as date_type
    if not content.strip():
        return "内容不能为空"
    entry_date = date or date_type.today().isoformat()
    url = f"{SUPABASE_URL}/rest/v1/diary"
    payload = {"content": content.strip(), "date": entry_date}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers={**_supabase_headers(), "Prefer": "return=representation"}, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
    return f"日记已保存，日期：{entry_date}"


async def tool_list_diary(limit: int = 10, date_from: Optional[str] = None, date_to: Optional[str] = None) -> list:
    """查询日记"""
    url = f"{SUPABASE_URL}/rest/v1/diary"
    params: dict = {"select": "id,date,content,created_at", "order": "date.desc", "limit": str(limit)}
    if date_from:
        params["date"] = f"gte.{date_from}"
    if date_to:
        params["date"] = f"lte.{date_to}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
    return r.json()


async def tool_log_mood(mood: str, intensity: int, note: Optional[str] = None) -> str:
    """记录情绪"""
    if not mood.strip():
        return "mood 不能为空"
    intensity = max(1, min(5, int(intensity)))
    url = f"{SUPABASE_URL}/rest/v1/mood_logs"
    payload = {"mood": mood.strip(), "intensity": intensity, "note": note or ""}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers={**_supabase_headers(), "Prefer": "return=representation"}, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
    return f"情绪已记录：{mood}（强度 {intensity}/5）"


async def tool_get_mood_history(limit: int = 20, mood: Optional[str] = None) -> list:
    """查询情绪历史"""
    url = f"{SUPABASE_URL}/rest/v1/mood_logs"
    params: dict = {"select": "id,mood,intensity,note,created_at", "order": "created_at.desc", "limit": str(limit)}
    if mood:
        params["mood"] = f"eq.{mood}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
    return r.json()


async def tool_search_note(query: str, include_done: bool = False) -> list:
    """按关键词搜索 notes"""
    url = f"{SUPABASE_URL}/rest/v1/notes"
    params: dict = {
        "select": "id,content,done,created_at",
        "content": f"ilike.*{query}*",
        "order": "created_at.desc",
        "limit": "30"
    }
    if not include_done:
        params["done"] = "eq.false"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
    return r.json()


async def tool_get_memory_stats() -> dict:
    """记忆库统计"""
    headers = _supabase_headers()
    async with httpx.AsyncClient(timeout=15) as client:
        # 总数 & 按分类
        r_total = await client.get(
            f"{SUPABASE_URL}/rest/v1/memories",
            headers={**headers, "Prefer": "count=exact"},
            params={"select": "id", "forgotten": "eq.false", "limit": "1"}
        )
        total = int(r_total.headers.get("content-range", "0/0").split("/")[-1]) if r_total.status_code < 400 else 0

        # 权重分布
        r_weights = await client.get(
            f"{SUPABASE_URL}/rest/v1/memories",
            headers=headers,
            params={"select": "weight,category", "forgotten": "eq.false", "limit": "1000"}
        )
        rows = r_weights.json() if r_weights.status_code < 400 else []

    weight_buckets = {"高(>0.8)": 0, "中(0.5-0.8)": 0, "低(0.2-0.5)": 0, "濒危(<0.2)": 0}
    category_count: dict = {}
    at_risk = []

    for row in rows:
        w = row.get("weight", 0.6)
        cat = row.get("category") or "未分类"
        category_count[cat] = category_count.get(cat, 0) + 1
        if w > 0.8:
            weight_buckets["高(>0.8)"] += 1
        elif w >= 0.5:
            weight_buckets["中(0.5-0.8)"] += 1
        elif w >= 0.2:
            weight_buckets["低(0.2-0.5)"] += 1
        else:
            weight_buckets["濒危(<0.2)"] += 1
            at_risk.append({"weight": round(w, 3), "category": cat})

    return {
        "total": total,
        "weight_distribution": weight_buckets,
        "by_category": category_count,
        "at_risk_count": len(at_risk),
        "at_risk_samples": at_risk[:5]
    }


async def tool_get_weather(city: str, forecast: bool = False) -> dict:
    """直接用城市名查天气，内部自动 geocode"""
    # 1. geocode
    geo = await amap_geocode(city)
    if geo.get("status") != "1" or not geo.get("geocodes"):
        return {"error": f"找不到城市：{city}"}
    adcode = geo["geocodes"][0].get("adcode", "")
    if not adcode:
        return {"error": "geocode 未返回 adcode"}
    # 2. 查天气
    extensions = "all" if forecast else "base"
    weather = await amap_weather(adcode, extensions=extensions)
    return weather


async def tool_generate_image(prompt: str, size: str = "1024x1024") -> dict:
    """调用硅基流动 FLUX.1-schnell 生成图片，返回图片URL"""
    if not SILICONFLOW_KEY:
        return {"error": "未配置 SILICONFLOW_KEY，请在 Railway 环境变量里添加"}
    if not prompt:
        return {"error": "prompt 不能为空"}
    url = "https://api.siliconflow.cn/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "image_size": size,
        "num_inference_steps": 4,
        "batch_size": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            return {"error": f"硅基流动返回错误 {r.status_code}: {r.text[:300]}"}
        data = r.json()
        image_url = data["images"][0]["url"]
        return {
            "image_url": image_url,
            "message": f"图片已生成！点击链接查看：{image_url}",
            "prompt": prompt,
        }
    except Exception as e:
        return {"error": f"生图失败：{e}"}


# ============================================================
# generate_card 卡片生成器
# ============================================================

def tool_generate_card(card_type: str, title: str, body: str, footer: str = "", accent: str = "") -> str:
    """生成 HTML 卡片，返回完整 HTML 字符串"""

    # 各类型主题配置
    themes = {
        "anniversary": {
            "bg": "linear-gradient(135deg, #0f0c29, #302b63, #24243e)",
            "card_bg": "rgba(255,255,255,0.07)",
            "border": "rgba(255,215,100,0.3)",
            "title_color": "#ffd700",
            "body_color": "#e8e0ff",
            "footer_color": "rgba(255,215,100,0.6)",
            "deco": "✦",
            "deco_color": "#ffd700",
            "shadow": "0 8px 32px rgba(255,215,100,0.15)",
        },
        "love_letter": {
            "bg": "linear-gradient(135deg, #fff0f6, #ffe4ee, #ffd6e7)",
            "card_bg": "rgba(255,255,255,0.85)",
            "border": "rgba(255,150,180,0.4)",
            "title_color": "#d63384",
            "body_color": "#5c2d44",
            "footer_color": "#f48cb0",
            "deco": "♡",
            "deco_color": "#ff85a1",
            "shadow": "0 8px 32px rgba(255,100,150,0.18)",
        },
        "custom": {
            "bg": "linear-gradient(135deg, #e0f2fe, #f0fdf4, #fdf4ff)",
            "card_bg": "rgba(255,255,255,0.82)",
            "border": "rgba(150,200,255,0.4)",
            "title_color": accent or "#6366f1",
            "body_color": "#334155",
            "footer_color": accent or "#94a3b8",
            "deco": "✿",
            "deco_color": accent or "#a78bfa",
            "shadow": "0 8px 32px rgba(100,100,255,0.12)",
        },
    }

    t = themes.get(card_type, themes["custom"])
    footer_html = f'<div class="footer">{footer}</div>' if footer else ""

    # body 换行处理
    body_lines = body.replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: {t['bg']};
    font-family: 'PingFang SC', 'Noto Serif SC', serif;
    padding: 24px;
  }}
  .card {{
    background: {t['card_bg']};
    border: 1px solid {t['border']};
    border-radius: 20px;
    padding: 40px 36px;
    max-width: 360px;
    width: 100%;
    box-shadow: {t['shadow']};
    backdrop-filter: blur(12px);
    text-align: center;
    position: relative;
  }}
  .deco {{
    font-size: 28px;
    color: {t['deco_color']};
    margin-bottom: 16px;
    letter-spacing: 8px;
    opacity: 0.9;
  }}
  .title {{
    font-size: 22px;
    font-weight: 700;
    color: {t['title_color']};
    margin-bottom: 20px;
    line-height: 1.4;
  }}
  .divider {{
    width: 48px;
    height: 2px;
    background: {t['border']};
    margin: 0 auto 20px;
    border-radius: 2px;
  }}
  .body {{
    font-size: 15px;
    color: {t['body_color']};
    line-height: 1.9;
    margin-bottom: 24px;
  }}
  .footer {{
    font-size: 13px;
    color: {t['footer_color']};
    margin-top: 8px;
    opacity: 0.85;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="deco">{t['deco']} {t['deco']} {t['deco']}</div>
    <div class="title">{title}</div>
    <div class="divider"></div>
    <div class="body">{body_lines}</div>
    {footer_html}
  </div>
</body>
</html>"""

    return html


async def cleanup_old_mood_logs(days: int = 90) -> dict:
    """删除超过 N 天的情绪记录"""
    url = f"{SUPABASE_URL}/rest/v1/mood_logs"
    cutoff = (datetime.now(SH_TZ) - timedelta(days=days)).isoformat()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(
            url,
            headers={**_supabase_headers(), "Prefer": "return=representation"},
            params={"created_at": f"lt.{cutoff}"}
        )
    if r.status_code >= 400:
        return {"error": f"Supabase {r.status_code}: {r.text[:100]}"}
    deleted = len(r.json()) if r.text else 0
    return {"deleted_mood_logs": deleted, "older_than_days": days}


async def tool_get_phone_status() -> dict:
    """查询最近几条手机状态"""
    url = f"{SUPABASE_URL}/rest/v1/phone_status"
    params = {
        "select": "status_text,created_at",
        "order": "created_at.desc",
        "limit": "5"
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_supabase_headers(), params=params)

    if r.status_code >= 400:
        return {"error": f"Supabase {r.status_code}"}
    rows = r.json()
    if not rows:
        return {"error": "暂无状态记录，MacroDroid 可能还未上报"}

    rows = list(reversed(rows))  # 改为旧→新，最后一条是最新状态

    from datetime import timezone as _tz
    results = []
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            sh_now = datetime.now(_tz(timedelta(hours=8)))
            diff = int((sh_now - dt.astimezone(_tz(timedelta(hours=8)))).total_seconds() / 60)
            time_hint = f"{diff}分钟前" if diff > 0 else "刚刚"
            results.append(f"[{time_hint}] {row['status_text']}")
        except Exception:
            results.append(row.get("status_text", ""))

    return {"records": results}
