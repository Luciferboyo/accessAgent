import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskRecord:
    task_id: str
    task: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    usage: Optional[dict] = None
    # 实时进度：执行中时更新，供外部轮询查看当前步骤
    progress: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None


class TaskStore:
    """线程安全的任务队列 + 状态存储（内存）"""

    def __init__(self):
        self.tasks: dict[str, TaskRecord] = {}
        self.queue: asyncio.Queue = asyncio.Queue()

    async def submit(self, task: str) -> TaskRecord:
        task_id = str(uuid.uuid4())[:8]
        record = TaskRecord(task_id=task_id, task=task)
        self.tasks[task_id] = record
        await self.queue.put(task_id)
        print(f"[TaskStore] 新任务入队 [{task_id}]：{task}")
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self.tasks.get(task_id)

    def list_all(self) -> list[TaskRecord]:
        return sorted(self.tasks.values(), key=lambda r: r.created_at, reverse=True)

    def update(self, task_id: str, **kwargs):
        record = self.tasks.get(task_id)
        if record:
            for k, v in kwargs.items():
                setattr(record, k, v)

    def delete(self, task_id: str) -> bool:
        """删除任务记录，返回是否成功删除"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
