from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from core.task_store import TaskStore, TaskRecord

# 全局 store，由 main.py 注入
_store: Optional[TaskStore] = None


def create_app(store: TaskStore) -> FastAPI:
    global _store
    _store = store

    app = FastAPI(
        title="AccessAgent API",
        description="手机自动化 Agent — HTTP 任务接口",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 请求 / 响应模型 ───────────────────────────────────

    class TaskRequest(BaseModel):
        task: str

    class TaskResponse(BaseModel):
        task_id: str
        task: str
        status: str
        result: Optional[str] = None
        error: Optional[str] = None
        created_at: str
        completed_at: Optional[str] = None
        usage: Optional[dict] = None

    # ── 工具函数 ──────────────────────────────────────────

    def to_resp(record: TaskRecord) -> TaskResponse:
        return TaskResponse(
            task_id=record.task_id,
            task=record.task,
            status=record.status,
            result=record.result,
            error=record.error,
            created_at=record.created_at,
            completed_at=record.completed_at,
            usage=record.usage,
        )

    # ── 路由 ─────────────────────────────────────────────

    @app.post("/task", response_model=TaskResponse, summary="提交新任务")
    def submit_task(req: TaskRequest):
        """
        提交一个自动化任务。
        任务会进入队列，等待 Android 设备连接后依次执行。

        返回 `task_id`，可用于后续查询状态。
        """
        record = _store.submit(req.task)
        return to_resp(record)

    @app.get("/task/{task_id}", response_model=TaskResponse, summary="查询任务状态")
    def get_task(task_id: str):
        """
        通过 task_id 查询任务的当前状态和执行结果。

        - `pending`：排队等待
        - `running`：执行中
        - `completed`：成功完成，`result` 字段含结果
        - `failed`：失败，`error` 字段含原因
        """
        record = _store.get(task_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        return to_resp(record)

    @app.get("/tasks", response_model=list[TaskResponse], summary="列出所有任务")
    def list_tasks():
        """返回所有任务列表（按创建时间倒序）。"""
        return [to_resp(r) for r in _store.list_all()]

    @app.delete("/task/{task_id}", summary="删除任务记录")
    def delete_task(task_id: str):
        """删除任务记录（仅限已完成或失败的任务）。"""
        record = _store.get(task_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        if record.status in ("pending", "running"):
            raise HTTPException(status_code=400, detail="任务正在执行中，无法删除")
        del _store.tasks[task_id]
        return {"message": f"任务 {task_id} 已删除"}

    @app.get("/health", summary="健康检查")
    def health():
        pending = sum(1 for r in _store.tasks.values() if r.status == "pending")
        running = sum(1 for r in _store.tasks.values() if r.status == "running")
        return {
            "status": "ok",
            "queue_pending": pending,
            "running": running,
            "total_tasks": len(_store.tasks),
        }

    return app
