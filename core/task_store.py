import asyncio
import uuid
from dataclasses import dataclass, field, fields
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
    # 用户自定义最大步数（None 时使用 config.MAX_STEPS 全局默认值）
    max_steps: Optional[int] = None


# 允许通过 update(**kwargs) 写入的字段白名单，避免拼写错误静默写入非法属性
_ALLOWED_UPDATE_FIELDS = frozenset(f.name for f in fields(TaskRecord))


class TaskStore:
    """线程/协程安全的任务队列 + 状态存储（内存）"""

    def __init__(self):
        self.tasks: dict[str, TaskRecord] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        # 保护 self.tasks 的并发访问（FastAPI 端点与 WS handler 同时读写）
        self._lock: asyncio.Lock = asyncio.Lock()

    async def submit(self, task: str, max_steps: Optional[int] = None) -> TaskRecord:
        task_id = str(uuid.uuid4())[:8]
        record = TaskRecord(task_id=task_id, task=task, max_steps=max_steps)
        async with self._lock:
            self.tasks[task_id] = record
        await self.queue.put(task_id)
        steps_info = f"，max_steps={max_steps}" if max_steps else ""
        print(f"[TaskStore] 新任务入队 [{task_id}]：{task}{steps_info}")
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        # dict.get 自身是 GIL 原子操作，无需加锁
        return self.tasks.get(task_id)

    def snapshot(self) -> list[TaskRecord]:
        """返回 tasks 的浅拷贝列表，避免迭代过程中被其他协程修改导致 RuntimeError。"""
        return list(self.tasks.values())

    def list_all(self) -> list[TaskRecord]:
        return sorted(self.snapshot(), key=lambda r: r.created_at, reverse=True)

    def update(self, task_id: str, **kwargs):
        """
        更新任务记录字段。
        - 仅接受 TaskRecord 已定义的字段；非法字段会触发警告并被丢弃，避免拼写错误静默生效
        - 单次调用是同步 GIL 原子序列，对 dataclass 字段赋值是非原子但安全（无对象重建）
        """
        record = self.tasks.get(task_id)
        if record is None:
            return
        invalid = [k for k in kwargs if k not in _ALLOWED_UPDATE_FIELDS]
        if invalid:
            print(f"[TaskStore] 警告：update() 忽略未知字段 {invalid}（task_id={task_id}）")
        for k, v in kwargs.items():
            if k in _ALLOWED_UPDATE_FIELDS:
                setattr(record, k, v)

    def delete(self, task_id: str) -> bool:
        """删除任务记录，返回是否成功删除"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False

    async def requeue(self, task_id: str) -> bool:
        """
        将任务重新放回队列尾部（用于 WebSocket 连接中断、任务未真正开始等场景）。
        若任务已不存在则返回 False。
        """
        if task_id not in self.tasks:
            return False
        record = self.tasks[task_id]
        record.status = TaskStatus.PENDING
        record.error = None
        record.completed_at = None
        record.progress = None
        record.current_step = None
        record.total_steps = None
        await self.queue.put(task_id)
        print(f"[TaskStore] 任务重新入队 [{task_id}]：{record.task[:40]}")
        return True
