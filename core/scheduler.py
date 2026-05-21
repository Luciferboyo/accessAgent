"""
定时任务调度器（Scheduler）
============================
基于 APScheduler.AsyncIOScheduler，将 ScheduleStore 中的 cron 配置注册到事件循环，
到点自动调用 TaskStore.submit()，并按用户配置做节假日跳过 + 失败重试。

设计要点：
- AsyncIOScheduler 共用主进程的 asyncio 事件循环，无需多进程/多线程
- 每条 schedule 对应一个 APScheduler Job，job_id = schedule.id
- 触发时先做节假日校验，再 submit 任务并等待结果，失败按用户配置重试
"""

import asyncio
from datetime import date
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.holiday import get_day_type
from core.schedule_store import ScheduleStore
from core.task_store import TaskStore, TaskStatus


# 等待任务执行完成的最长时间（单次提交）；与 config.MAX_STEPS 配套，
# 一般任务在 5-10 分钟内能跑完，给到 30 分钟兜底。
DEFAULT_WAIT_TIMEOUT_SECONDS = 30 * 60
WAIT_POLL_SECONDS = 5


class Scheduler:
    def __init__(self,
                 task_store: TaskStore,
                 schedule_store: ScheduleStore,
                 timezone: str = "Asia/Shanghai"):
        self.task_store = task_store
        self.schedule_store = schedule_store
        self.scheduler = AsyncIOScheduler(timezone=timezone)
        self._started = False

    # ── 生命周期 ──────────────────────────────────────────────────────

    def start(self) -> None:
        """启动调度器并加载所有 enabled schedule。"""
        if self._started:
            return
        self.scheduler.start()
        self._started = True
        loaded = 0
        for sched in self.schedule_store.list():
            if sched.get("enabled"):
                try:
                    self._register_job(sched)
                    loaded += 1
                except Exception as e:
                    print(f"[Scheduler] 注册 {sched['id']} 失败：{e}")
        print(f"[Scheduler] 已启动，加载 {loaded} 个定时任务")

    def shutdown(self) -> None:
        if not self._started:
            return
        try:
            self.scheduler.shutdown(wait=False)
        except Exception as e:
            print(f"[Scheduler] 关闭异常：{e}")
        self._started = False

    # ── Job 注册 ──────────────────────────────────────────────────────

    @staticmethod
    def validate_cron(expr: str) -> Optional[str]:
        """
        校验 cron 表达式。合法返回 None，非法返回错误信息字符串。
        """
        try:
            CronTrigger.from_crontab(expr)
            return None
        except Exception as e:
            return f"无效的 cron 表达式：{e}"

    def _register_job(self, sched: dict) -> None:
        trigger = CronTrigger.from_crontab(sched["cron"], timezone=self.scheduler.timezone)
        self.scheduler.add_job(
            self._fire,
            trigger=trigger,
            args=[sched["id"]],
            id=sched["id"],
            replace_existing=True,
            misfire_grace_time=300,   # 错过 5 分钟内仍执行
            coalesce=True,            # 多次错过合并为一次
        )

    def _unregister_job(self, schedule_id: str) -> None:
        try:
            self.scheduler.remove_job(schedule_id)
        except Exception:
            pass   # 不存在就忽略

    def get_next_run(self, schedule_id: str) -> Optional[str]:
        """返回下次触发时间的 ISO 字符串。"""
        try:
            job = self.scheduler.get_job(schedule_id)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        except Exception:
            pass
        return None

    # ── CRUD 包装：自动同步 store 与 scheduler ─────────────────────────

    def add_schedule(self, data: dict) -> dict:
        err = self.validate_cron(data["cron"])
        if err:
            raise ValueError(err)
        rec = self.schedule_store.create(data)
        if rec.get("enabled") and self._started:
            self._register_job(rec)
        return rec

    def update_schedule(self, schedule_id: str, **kwargs) -> Optional[dict]:
        if "cron" in kwargs:
            err = self.validate_cron(kwargs["cron"])
            if err:
                raise ValueError(err)
        rec = self.schedule_store.update(schedule_id, **kwargs)
        if rec is None:
            return None
        # 重新注册以反映新的 cron / enabled 状态
        self._unregister_job(schedule_id)
        if rec.get("enabled") and self._started:
            self._register_job(rec)
        return rec

    def delete_schedule(self, schedule_id: str) -> bool:
        self._unregister_job(schedule_id)
        return self.schedule_store.delete(schedule_id)

    # ── 触发逻辑（cron 到点时被 APScheduler 调用）─────────────────────

    async def _fire(self, schedule_id: str) -> None:
        """
        定时任务到点：
        1. 检查节假日，命中跳过策略则记一次 skip
        2. 否则提交任务到 TaskStore，等待执行完成
        3. 失败按 retry_max_attempts / retry_interval_seconds 重试
        """
        sched = self.schedule_store.get(schedule_id)
        if not sched or not sched.get("enabled"):
            print(f"[Scheduler] 跳过 {schedule_id}：未找到或已禁用")
            return

        # ── 节假日 / 调休 检查（按 schedule 配置的 provider + region） ──
        if sched.get("skip_holidays"):
            provider = sched.get("holiday_provider", "china_timor")
            region = sched.get("holiday_region", "")
            day_type = await get_day_type(date.today(), provider=provider, region=region)
            if day_type == "holiday":
                print(f"[Scheduler] {schedule_id} 今天是节假日（{provider}{':' + region if region else ''}），跳过")
                self.schedule_store.mark_run(schedule_id, "", "skipped_holiday")
                return
            if day_type == "makeup" and not sched.get("include_makeup_workdays", True):
                print(f"[Scheduler] {schedule_id} 今天是调休补班日（用户配置跳过），跳过")
                self.schedule_store.mark_run(schedule_id, "", "skipped_makeup")
                return
            if day_type == "weekend":
                print(f"[Scheduler] {schedule_id} 今天是周末（skip_holidays=true 时一并跳过），跳过")
                self.schedule_store.mark_run(schedule_id, "", "skipped_weekend")
                return

        max_attempts = max(1, int(sched.get("retry_max_attempts", 3)))
        retry_interval = max(0, int(sched.get("retry_interval_seconds", 300)))
        last_task_id = ""
        last_status = "failed"

        for attempt in range(1, max_attempts + 1):
            record = await self.task_store.submit(
                sched["task"], max_steps=sched.get("max_steps")
            )
            last_task_id = record.task_id
            print(f"[Scheduler] {schedule_id} 第 {attempt}/{max_attempts} 次触发，"
                  f"task_id={record.task_id}")

            status = await self._wait_completion(record.task_id)
            last_status = status

            if status == TaskStatus.COMPLETED:
                print(f"[Scheduler] {schedule_id} 成功（第 {attempt} 次）")
                self.schedule_store.mark_run(schedule_id, last_task_id, "completed")
                return

            if attempt < max_attempts:
                print(f"[Scheduler] {schedule_id} 失败（{status}），{retry_interval}s 后重试")
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)

        print(f"[Scheduler] {schedule_id} 已连续 {max_attempts} 次失败，放弃")
        self.schedule_store.mark_run(schedule_id, last_task_id, f"failed_after_{max_attempts}_attempts")

    async def trigger_now(self, schedule_id: str,
                          bypass_holiday_check: bool = True) -> Optional[str]:
        """
        立即手动触发一次某个 schedule，返回新创建的 task_id。
        - bypass_holiday_check=True（默认）：忽略节假日设置直接执行，方便测试
        - 不会经过 retry_max_attempts 循环；用户手动触发就执行一次
        - 调度器自动安排的 cron 触发不受影响
        """
        sched = self.schedule_store.get(schedule_id)
        if sched is None:
            return None

        # 选择性跳过节假日检查
        if not bypass_holiday_check and sched.get("skip_holidays"):
            day_type = await get_day_type(
                date.today(),
                provider=sched.get("holiday_provider", "china_timor"),
                region=sched.get("holiday_region", ""),
            )
            if day_type in ("holiday", "weekend"):
                print(f"[Scheduler] trigger_now {schedule_id} 命中 {day_type}，跳过")
                return None
            if day_type == "makeup" and not sched.get("include_makeup_workdays", True):
                return None

        record = await self.task_store.submit(
            sched["task"], max_steps=sched.get("max_steps")
        )
        # 异步等待完成并 mark_run，不阻塞 trigger_now 返回
        asyncio.create_task(self._track_manual_trigger(schedule_id, record.task_id))
        print(f"[Scheduler] trigger_now {schedule_id} 已提交，task_id={record.task_id}")
        return record.task_id

    async def _track_manual_trigger(self, schedule_id: str, task_id: str) -> None:
        """后台等待手动触发的任务完成并写入 mark_run，不计入 retry 循环。"""
        status = await self._wait_completion(task_id)
        self.schedule_store.mark_run(schedule_id, task_id, f"manual_{status}")

    @staticmethod
    def _status_str(status) -> str:
        """统一拿到 'completed' / 'failed' 等小写字符串，无论 status 是 enum 还是字符串。"""
        if hasattr(status, "value"):
            return status.value
        return str(status)

    async def _wait_completion(self, task_id: str,
                               timeout: int = DEFAULT_WAIT_TIMEOUT_SECONDS) -> str:
        """轮询 TaskStore 直到任务到终态或超时。"""
        elapsed = 0
        while elapsed < timeout:
            rec = self.task_store.get(task_id)
            if rec is None:
                return "missing"
            if rec.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                              "completed", "failed"):
                return self._status_str(rec.status)
            await asyncio.sleep(WAIT_POLL_SECONDS)
            elapsed += WAIT_POLL_SECONDS
        return "timeout"
