import os
import tempfile

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import config
from core.task_store import TaskStore, TaskRecord, TaskStatus

# 全局 store / vision_llm，由 main.py 注入
_store: Optional[TaskStore] = None
_vision_llm = None


def _parse_origins(raw: str) -> list[str]:
    """解析 CORS_ORIGINS 字符串。'*' 直接放行；逗号分隔多个来源。"""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(store: TaskStore, vision_llm=None) -> FastAPI:
    global _store, _vision_llm
    _store = store
    _vision_llm = vision_llm

    app = FastAPI(
        title="AccessAgent API",
        description="手机自动化 Agent — HTTP 任务接口",
        version="1.0.0",
    )

    origins = _parse_origins(config.CORS_ORIGINS)
    # 仅当配置为 "*" 时才放行所有来源；其余情况严格按白名单
    allow_credentials = origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost"],
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Token"],
    )

    # ── 鉴权依赖（仅写操作使用）─────────────────────────
    def require_api_token(x_api_token: Optional[str] = Header(None)):
        """若配置了 API_TOKEN，则写操作必须携带正确的 X-API-Token 头。"""
        if config.API_TOKEN and x_api_token != config.API_TOKEN:
            raise HTTPException(status_code=401, detail="无效或缺失的 X-API-Token")
        return True

    # ── 请求 / 响应模型 ───────────────────────────────────

    class TaskRequest(BaseModel):
        task: str
        max_steps: Optional[int] = None  # 自定义最大步数，不传则使用服务器默认值

    class TaskResponse(BaseModel):
        task_id: str
        task: str
        status: str
        result: Optional[str] = None
        error: Optional[str] = None
        created_at: str
        completed_at: Optional[str] = None
        usage: Optional[dict] = None
        # 实时进度字段（running 状态时有值）
        progress: Optional[str] = None
        current_step: Optional[int] = None
        total_steps: Optional[int] = None
        max_steps: Optional[int] = None  # 该任务实际使用的最大步数限制

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
            progress=record.progress,
            current_step=record.current_step,
            total_steps=record.total_steps,
            max_steps=record.max_steps,
        )

    # ── 路由 ─────────────────────────────────────────────

    @app.post("/task", response_model=TaskResponse, summary="提交新任务",
              dependencies=[Depends(require_api_token)])
    async def submit_task(req: TaskRequest):
        """
        提交一个自动化任务。
        任务会进入队列，等待 Android 设备连接后依次执行。

        返回 `task_id`，可用于后续查询状态。
        """
        record = await _store.submit(req.task, max_steps=req.max_steps)
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

    @app.delete("/task/{task_id}", summary="删除任务记录",
                dependencies=[Depends(require_api_token)])
    def delete_task(task_id: str):
        """删除任务记录（仅限已完成或失败的任务）。"""
        record = _store.get(task_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        if record.status in (TaskStatus.PENDING, TaskStatus.RUNNING,
                             "pending", "running"):
            raise HTTPException(status_code=400, detail="任务正在执行中，无法删除")
        _store.delete(task_id)
        return {"message": f"任务 {task_id} 已删除"}

    @app.get("/health", summary="健康检查")
    def health():
        # 用 snapshot 取快照后再统计，避免与并发写入产生 dict iteration race
        snapshot = _store.snapshot()
        pending = sum(1 for r in snapshot
                      if r.status in (TaskStatus.PENDING, "pending"))
        running = sum(1 for r in snapshot
                      if r.status in (TaskStatus.RUNNING, "running"))
        return {
            "status": "ok",
            "queue_pending": pending,
            "running": running,
            "total_tasks": len(snapshot),
        }

    # ── 测试接口：直接上传图片文件 → VisionLLM ───────────────────────

    class VisionTestResponse(BaseModel):
        image_size_bytes: int
        image_size_kb: float
        image_size_mb: float
        llm_response: Optional[str] = None
        error: Optional[str] = None

    @app.post("/test/vision-raw", response_model=VisionTestResponse,
              summary="测试：上传截图文件发送给 VisionLLM",
              dependencies=[Depends(require_api_token)])
    async def test_vision_raw(
        file: UploadFile = File(..., description="截图文件（jpg/png/webp）"),
        prompt: str = Form("请描述这张图片的内容。"),
    ):
        """
        直接上传图片文件（不做任何压缩/缩放）发给视觉模型，
        返回图片容量和 LLM 响应，用于测试 API 能否接受大图（如 3 MB）。

        **使用方法（curl）**：
        ```
        curl -X POST http://localhost:8000/test/vision-raw \\
             -F "file=@/path/to/screenshot.jpg" \\
             -F "prompt=请描述这张图片"
        ```
        """
        if _vision_llm is None:
            raise HTTPException(status_code=503, detail="VisionLLM 未初始化，请检查服务启动配置")

        raw_bytes = await file.read()
        size_bytes = len(raw_bytes)
        size_kb = round(size_bytes / 1024, 2)
        size_mb = round(size_bytes / 1024 / 1024, 3)

        # 根据文件头推断扩展名，优先用上传文件名
        suffix = os.path.splitext(file.filename or "")[-1].lower() or ".jpg"
        if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
            suffix = ".jpg"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(raw_bytes)
            tmp.flush()
            tmp.close()

            llm_resp, _usage = _vision_llm.predict(prompt, tmp.name)
            return VisionTestResponse(
                image_size_bytes=size_bytes,
                image_size_kb=size_kb,
                image_size_mb=size_mb,
                llm_response=llm_resp,
            )
        except Exception as e:
            return VisionTestResponse(
                image_size_bytes=size_bytes,
                image_size_kb=size_kb,
                image_size_mb=size_mb,
                error=str(e),
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError as e:
                print(f"[/test/vision-raw] 临时文件清理失败：{tmp.name}（{e}）")

    return app
