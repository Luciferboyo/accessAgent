import asyncio
import uvicorn

from config import config
from core.task_store import TaskStore
from core.server import AccessAgentServer
from api.app import create_app


async def main():
    # 共享任务队列
    store = TaskStore()

    # FastAPI HTTP 服务（端口 8000）
    app = create_app(store)
    http_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",   # 减少 uvicorn 日志噪音
    )
    http_server = uvicorn.Server(http_config)

    # WebSocket 服务（端口 8765）
    ws_server = AccessAgentServer(store)

    print("=" * 50)
    print("🚀 AccessAgent 启动")
    print(f"   HTTP API : http://0.0.0.0:8000")
    print(f"   API 文档 : http://0.0.0.0:8000/docs")
    print(f"   WebSocket: ws://0.0.0.0:{config.PORT}")
    print("=" * 50)
    print("提交任务示例：")
    print('  curl -X POST http://localhost:8000/task \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"task": "搜索NBA季后赛赛程并汇报给我"}\'')
    print("=" * 50)

    # 同时运行两个服务
    await asyncio.gather(
        ws_server.start("0.0.0.0", config.PORT),
        http_server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
