import asyncio
import signal
import uvicorn

from config import config
from core.task_store import TaskStore
from core.server import AccessAgentServer
from core.schedule_store import ScheduleStore
from core.scheduler import Scheduler
from api.app import create_app


async def main():
    # 共享任务队列
    store = TaskStore()

    # WebSocket 服务（端口 8765）— 先创建，以便共享 vision_llm
    ws_server = AccessAgentServer(store)

    # 定时任务调度器（共用主事件循环）
    schedule_store = ScheduleStore()
    scheduler = Scheduler(store, schedule_store, timezone=config.SCHEDULER_TIMEZONE)

    # FastAPI HTTP 服务（端口 8000）
    app = create_app(store, vision_llm=ws_server.vision_llm, scheduler=scheduler)
    http_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",   # 减少 uvicorn 日志噪音
    )
    http_server = uvicorn.Server(http_config)

    print("=" * 50)
    print("🚀 AccessAgent 启动")
    print(f"   HTTP API : http://0.0.0.0:8000")
    print(f"   API 文档 : http://0.0.0.0:8000/docs")
    print(f"   WebSocket: ws://0.0.0.0:{config.PORT}")
    print(f"   时区     : {config.SCHEDULER_TIMEZONE}")
    print("=" * 50)
    print("提交任务示例：")
    print('  curl -X POST http://localhost:8000/task \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"task": "搜索NBA季后赛赛程并汇报给我"}\'')
    print("创建工作日打卡定时任务示例：")
    print('  curl -X POST http://localhost:8000/schedule \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"name":"早上打卡","task":"打开Lark点击Clock In打卡",'
          '"cron":"0 9 * * 1-5","skip_holidays":true}\'')
    print("=" * 50)

    # 启动调度器（在 uvicorn / ws_server 之前，确保已加载的 job 立即就绪）
    scheduler.start()

    # ── 优雅退出：SIGINT / SIGTERM 先让 uvicorn 正常停止 ──────────
    loop = asyncio.get_running_loop()

    def _shutdown():
        print("\n[AccessAgent] 收到退出信号，正在优雅关闭...")
        http_server.should_exit = True   # 通知 uvicorn 停止接受新请求并退出

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, OSError):
            # Windows 不支持 add_signal_handler（SIGINT 除外），忽略即可
            pass

    # 同时运行两个服务；uvicorn 退出后 gather 取消 ws_server
    try:
        await asyncio.gather(
            ws_server.start("0.0.0.0", config.PORT),
            http_server.serve(),
            return_exceptions=True,
        )
    finally:
        # 优雅关闭调度器：触发中的任务会跑完，新任务不再排程
        scheduler.shutdown()
    print("[AccessAgent] 已退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Windows 上 SIGINT 由此捕获（add_signal_handler 不支持）
        print("\n[AccessAgent] 已退出")
