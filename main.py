import os
import json
import asyncio
import uuid
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

#=========================================================
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
        "description": "创建定时推送任务。run_at 为 ISO 时间字符串（例如 2026-02-25T08:30:00+08:00）。repeat可选：none/daily/weekly/hourly。",
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
    yield sse_data({"type": "endpoint", "uri": f"message/{session_id}"})
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
async def handle_message_mcp(session_id: str, request: Request):
    return await _handle_msg(session_id, request)


@app.post("/message/{session_id}")
async def handle_message_root(session_id: str, request: Request):
    return await _handle_msg(session_id, request)


async def _handle_msg(session_id: str, request: Request):
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
async def pushplus_notify(title: str, content: str, template: str = "txt"):
    if not PUSHPLUS_TOKEN:
        raise RuntimeError("PUSHPLUS_TOKEN missing")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post("https://www.pushplus.plus/send", json={
            "token": PUSHPLUS_TOKEN, "title": title,
            "content": content, "template": template, "channel": "app",
        })
        r.raise_for_status()
        return r.json()


# ---------- 定时推送 ----------
def _parse_run_at(run_at: str) -> str:
    s = run_at.strip()
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    if s.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", s):
        return s
    return s + "+08:00"


def _next_run_iso(prev_run_iso: str, repeat: str, now_iso_utc: str) -> Optional[str]:
    try:
        prev_dt = datetime.fromisoformat(prev_run_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        now_dt = datetime.fromisoformat(now_iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
    prev_local = prev_dt.astimezone(SH_TZ)
    now_local = now_dt.astimezone(SH_TZ)
    if repeat == "hourly":
        cand = now_local.replace(minute=prev_local.minute, second=0, microsecond=0)
        if cand <= now_local:
            cand += timedelta(hours=1)
    elif repeat == "daily":
        cand = now_local.replace(hour=prev_local.hour, minute=prev_local.minute, second=0, microsecond=0)
        if cand <= now_local:
            cand += timedelta(days=1)
    elif repeat == "weekly":
        cand = now_local.replace(hour=prev_local.hour, minute=prev_local.minute, second=0, microsecond=0)
        days_ahead = (prev_local.weekday() - cand.weekday()) % 7
        if days_ahead == 0 and cand <= now_local:
            days_ahead = 7
        cand += timedelta(days=days_ahead)
    else:
        return None
    return cand.astimezone(timezone.utc).isoformat()


async def create_push_schedule(*, title: str, content: str, run_at: str, repeat: str = "none") -> dict:
    payload = {
        "title": title, "content": content,
        "run_at": _parse_run_at(run_at),
        "repeat": repeat, "template": "txt", "enabled": True,
    }
    url = f"{SUPABASE_URL}/rest/v1/push_schedules"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers={**_supabase_headers(), "Prefer": "return=representation"}, json=payload)
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else payload


# ---------- 定时任务：每分钟触发 ----------
async def run_due_push_schedules() -> dict:
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_iso = now_utc.isoformat().replace("+00:00", "Z")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/push_schedules",
                headers=_supabase_headers(),
                params={
                    "select": "*", "enabled": "eq.true",
                    "run_at": f"lte.{now_iso}",
                    "or": "(status.eq.pending,status.is.null)",
                    "order": "run_at.asc", "limit": "50",
                },
            )
            if r.status_code >= 400:
                return {"ok": False, "error": f"fetch {r.status_code}"}
            jobs = r.json() or []except Exception as e:
        return {"ok": False, "error": str(e)}

    sent = 0
    for job in jobs:
        job_id = job.get("id")
        if not job_id:
            continue
        try:
            await pushplus_notify(title=job.get("title", "提醒"), content=job.get("content", ""))
            sent += 1
        except Exception:
            continuerepeat = (job.get("repeat") or "none").lower()
        patch: Dict[str, Any] = {"last_run_at": now_iso, "last_error": None}
        if repeat != "none":
            next_run = _next_run_iso(job.get("run_at", now_iso), repeat, now_iso)
            if next_run:
                patch.update({"run_at": next_run, "status": "pending", "enabled": True})
            else:
                patch.update({"enabled": False, "status": "error"})
        else:
            patch.update({"enabled": False, "status": "sent"})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/push_schedules?id=eq.{job_id}",
                    headers={**_supabase_headers(), "Prefer": "return=minimal"},
                    json=patch,
                )
        except Exception:
            pass

    return {"ok": True, "checked": len(jobs), "sent": sent}


@app.get("/cron/tick")
async def cron_tick(request: Request):
    if CRON_SECRET:
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if token != CRON_SECRET:
            raise HTTPException(status_code=401, detail="bad secret")
    return await run_due_push_schedules()
