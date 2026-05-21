"""
定时任务持久化（Schedule Store）
================================
将用户配置的定时任务存到 ./memory/schedules.json，进程重启后自动恢复。

字段说明（每条 schedule）：
  id                       8 位 hex，主键
  name                     人类可读的名字
  task                     提交给 Agent 的任务描述（同 POST /task 的 task 字段）
  cron                     5 字段标准 cron 表达式（分 时 日 月 周）
  max_steps                可选，单次任务最大步数
  skip_holidays            true 时跳过法定节假日
  include_makeup_workdays  true 时调休补班日仍执行（默认 true）
  retry_max_attempts       任务失败时重试次数（含首次，默认 3）
  retry_interval_seconds   重试间隔（默认 300s）
  enabled                  是否启用
  created_at / updated_at  ISO 时间戳
  last_run_at              上次触发时间
  last_task_id             上次触发的 task_id（可拿去查 /task/{id}）
  last_status              上次最终状态（completed/failed/skipped_holiday/skipped_makeup）
  run_count                历史触发总次数
"""

import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Optional


SCHEDULES_FILE = "./memory/schedules.json"


class ScheduleStore:
    def __init__(self, path: str = SCHEDULES_FILE):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.schedules: dict[str, dict] = self._load()

    # ── 持久化 ──────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError) as e:
                print(f"[ScheduleStore] 读取失败（{e}），重置为空")
        return {}

    def _save(self) -> None:
        dir_name = os.path.dirname(os.path.abspath(self.path))
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.schedules, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[ScheduleStore] 保存失败（{e}）")

    # ── CRUD ────────────────────────────────────────────────────────

    def list(self) -> list[dict]:
        return sorted(self.schedules.values(),
                      key=lambda r: r.get("created_at", ""))

    def get(self, schedule_id: str) -> Optional[dict]:
        return self.schedules.get(schedule_id)

    def create(self, data: dict) -> dict:
        schedule_id = data.get("id") or uuid.uuid4().hex[:8]
        now = datetime.now().isoformat()
        record = {
            "id": schedule_id,
            "name": data.get("name", ""),
            "task": data["task"],
            "cron": data["cron"],
            "max_steps": data.get("max_steps"),
            "skip_holidays": bool(data.get("skip_holidays", False)),
            "include_makeup_workdays": bool(data.get("include_makeup_workdays", True)),
            "retry_max_attempts": int(data.get("retry_max_attempts", 3)),
            "retry_interval_seconds": int(data.get("retry_interval_seconds", 300)),
            "enabled": bool(data.get("enabled", True)),
            "created_at": now,
            "updated_at": now,
            "last_run_at": None,
            "last_task_id": None,
            "last_status": None,
            "run_count": 0,
        }
        self.schedules[schedule_id] = record
        self._save()
        return record

    # 允许通过 update() 写入的字段白名单（与 task_store 同思路）
    _UPDATABLE = frozenset({
        "name", "task", "cron", "max_steps", "skip_holidays",
        "include_makeup_workdays", "retry_max_attempts",
        "retry_interval_seconds", "enabled",
    })

    def update(self, schedule_id: str, **kwargs) -> Optional[dict]:
        rec = self.schedules.get(schedule_id)
        if rec is None:
            return None
        invalid = [k for k in kwargs if k not in self._UPDATABLE]
        if invalid:
            print(f"[ScheduleStore] 忽略未知字段 {invalid}")
        for k, v in kwargs.items():
            if k in self._UPDATABLE:
                rec[k] = v
        rec["updated_at"] = datetime.now().isoformat()
        self._save()
        return rec

    def delete(self, schedule_id: str) -> bool:
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save()
            return True
        return False

    # ── 运行时统计 ───────────────────────────────────────────────────

    def mark_run(self, schedule_id: str, task_id: str, status: str) -> None:
        """触发后调用，记录本次执行结果。"""
        rec = self.schedules.get(schedule_id)
        if rec is None:
            return
        rec["last_run_at"] = datetime.now().isoformat()
        rec["last_task_id"] = task_id or None
        rec["last_status"] = status
        rec["run_count"] = rec.get("run_count", 0) + 1
        self._save()
