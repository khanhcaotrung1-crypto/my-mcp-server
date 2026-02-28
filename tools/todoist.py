import os
import httpx

TODOIST_API_TOKEN = os.getenv("TODOIST_API_TOKEN")
BASE_URL = "https://api.todoist.com/rest/v2"

headers = {
    "Authorization": f"Bearer {TODOIST_API_TOKEN}",
    "Content-Type": "application/json"
}

async def get_tasks():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/tasks", headers=headers)
        resp.raise_for_status()
        return resp.json()

async def create_task(content, due_string=None):
    data = {
        "content": content,
    }
    if due_string:
        data["due_string"] = due_string

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/tasks", headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()

async def complete_task(task_id):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/tasks/{task_id}/close",
            headers=headers
        )
        resp.raise_for_status()
        return {"status": "completed"}
