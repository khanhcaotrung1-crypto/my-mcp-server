"""
RikkaHub MCP Server
一个简单的 MCP 服务器，提供记忆管理和搜索功能
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import httpx

app = FastAPI(title="RikkaHub MCP Server")
from fastapi.responses import StreamingResponse
import asyncio, json

TOOLS = [
    {"name": "save_memory", "description": "保存一条记忆到数据库", "input_schema": {"type":"object","properties":{"title":{"type":"string"},"content":{"type":"string"},"category":{"type":"string"},"importance":{"type":"integer"}}, "required":["title","content"]}},
    {"name": "search_memory", "description": "搜索相关的记忆", "input_schema": {"type":"object","properties":{"query":{"type":"string"}}, "required":["query"]}},
    {"name": "get_recent_memories", "description": "获取最近的记忆", "input_schema": {"type":"object","properties":{"limit":{"type":"integer"}}}},
]

async def event_stream():
    yield f"data: {json.dumps({'type': 'init', 'tools': TOOLS})}\n\n"
    while True:
        await asyncio.sleep(10)
        yield f"data: {json.dumps({'type': 'ping'})}\n\n"

@app.get("/mcp")
async def mcp():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 环境变量
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MCP 工具定义
TOOLS = [
    {
        "name": "save_memory",
        "description": "保存一条记忆到数据库",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "记忆标题"},
                "content": {"type": "string", "description": "记忆内容"},
                "category": {"type": "string", "description": "分类（可选）"},
                "importance": {"type": "integer", "description": "重要性 1-5（可选）"}
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "search_memory",
        "description": "搜索相关的记忆",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_recent_memories",
        "description": "获取最近的记忆",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "返回数量，默认10"}
            }
        }
    }
]

async def save_memory_to_supabase(title: str, content: str, category: str = None, importance: int = 3):
    """保存记忆到 Supabase"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/memories",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            },
            json={
                "title": title,
                "content": content,
                "category": category,
                "importance": importance,
                "created_at": datetime.now().isoformat()
            }
        )
        return response.json()

async def search_memories_from_supabase(query: str):
    """从 Supabase 搜索记忆"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/memories",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            },
            params={
                "content": f"ilike.%{query}%",
                "order": "created_at.desc",
                "limit": 10
            }
        )
        return response.json()

async def get_recent_memories_from_supabase(limit: int = 10):
    """获取最近的记忆"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/memories",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            },
            params={
                "order": "created_at.desc",
                "limit": limit
            }
        )
        return response.json()

async def call_tool(tool_name: str, arguments: Dict[str, Any]):
    """调用工具"""
    if tool_name == "save_memory":
        result = await save_memory_to_supabase(
            title=arguments.get("title"),
            content=arguments.get("content"),
            category=arguments.get("category"),
            importance=arguments.get("importance", 3)
        )
        return {"success": True, "data": result}
    
    elif tool_name == "search_memory":
        result = await search_memories_from_supabase(arguments.get("query"))
        return {"success": True, "memories": result}
    
    elif tool_name == "get_recent_memories":
        result = await get_recent_memories_from_supabase(arguments.get("limit", 10))
        return {"success": True, "memories": result}
    
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

async def event_stream():
    """SSE 事件流"""
    # 发送初始化消息
    yield f"data: {json.dumps({'type': 'init', 'tools': TOOLS})}\n\n"
    
    # 保持连接
    while True:
        await asyncio.sleep(10)
        yield f"data: {json.dumps({'type': 'ping'})}\n\n"

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "RikkaHub MCP Server is running"}

@app.get("/mcp")
async def mcp_endpoint():
    """MCP SSE 端点"""
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/mcp/tools")
async def execute_tool(request: Request):
    """执行工具调用"""
    data = await request.json()
    tool_name = data.get("name")
    arguments = data.get("arguments", {})
    
    result = await call_tool(tool_name, arguments)
    return JSONResponse(content=result)

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return "ok"
