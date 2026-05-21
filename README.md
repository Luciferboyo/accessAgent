# AccessAgent

> 🌐 **[中文](#中文) | [English](#english)**

---

## 中文

基于多 Agent 架构的 Android 手机自动化框架，结合无障碍服务（Accessibility Service）与视觉 AI，通过自然语言驱动手机完成复杂任务。

### ✨ 特性

- **自然语言下发任务**：通过 HTTP API 提交任务，无需编写脚本
- **多 Agent 协作**：Planner 规划、Executor 决策、Reflector 反思，三者分工协作
- **文字优先 + 按需截图**:优先使用无障碍树做决策，仅在必要时调用视觉模型，节省成本
- **双模型架构**：文本决策用轻量 LLM（DeepSeek），视觉分析用多模态 LLM（Qwen-VL / GPT-4o）
- **三层任务记忆**：经验池（通用规则）+ 任务记忆（完整流程复用）+ 步骤片段记忆（跨任务子步骤复用）
- **🆕 定时任务**：cron 表达式 + 节假日跳过 + 失败重试，开箱即用的打卡场景
- **🆕 任务取消 / 手动触发 / 实时进度**：完整的运维 API
- **Token 用量统计**：每步打印 token 消耗与费用，任务结束汇总

### 🏗️ 架构

```
用户 / 调用方
    │  POST /task / POST /schedule
    ▼
┌─────────────────────────────────────────┐
│           PC 服务端 (main.py)            │
│                                         │
│  FastAPI HTTP (:8000)                   │
│  ┌─────────────────────────────────┐   │
│  │ TaskStore  ←——  Scheduler (cron) │   │
│  └────────────────┬────────────────┘   │
│                   │                     │
│  WebSocket (:8765)│                     │
│  ┌────────────────▼────────────────┐   │
│  │      AccessAgentServer 主循环     │   │
│  │  Planner → Executor → Reflector  │   │
│  └─────────────────────────────────┘   │
└────────────────────┬────────────────────┘
                     │ WebSocket
          ┌──────────▼──────────┐
          │    Android 设备      │
          │  无障碍 App / ADB    │
          └─────────────────────┘
```

### 🚀 快速开始

#### 1. 克隆 & 安装依赖

```bash
git clone https://github.com/boyovance/accessAgent.git
cd accessAgent
pip install -r requirements.txt
```

#### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填写：

```ini
TEXT_API_KEY=your_deepseek_api_key
TEXT_BASE_URL=https://api.deepseek.com/v1
TEXT_MODEL=deepseek-chat

VISION_API_KEY=your_vision_api_key
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_MODEL=qwen-vl-plus

# 可选：HTTP 安全
CORS_ORIGINS=http://localhost:3000
API_TOKEN=your_random_token            # 配置后写操作需 X-API-Token 头
SCHEDULER_TIMEZONE=Asia/Shanghai       # 定时任务时区
```

#### 3. 启动服务

```bash
python main.py
```

- HTTP API：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- WebSocket：`ws://localhost:8765`

#### 4. 连接 Android 设备（开发期用 mock_phone）

修改 `mock_phone.py` 顶部的 `DEVICE` 和 `ADB_PATH`，然后：

```bash
python mock_phone.py
```

#### 5. 提交任务

```bash
curl -X POST http://localhost:8000/task \
     -H "Content-Type: application/json" \
     -d '{"task": "帮我查询今天北京的天气"}'
```

### ⏰ 定时任务（含打卡场景）

**创建工作日早上 9:00 自动打卡，跳过法定节假日：**

```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "早上打卡",
    "task": "打开 Lark，进入考勤打卡页面，点击 Clock In 完成上班打卡",
    "cron": "0 9 * * 1-5",
    "skip_holidays": true,
    "holiday_provider": "china_timor",
    "include_makeup_workdays": true,
    "retry_max_attempts": 3,
    "retry_interval_seconds": 300
  }'
```

**国际地区（如美国节假日）：**

```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "US morning clock-in",
    "task": "Open the attendance app and clock in",
    "cron": "0 9 * * 1-5",
    "skip_holidays": true,
    "holiday_provider": "nager",
    "holiday_region": "US"
  }'
```

**节假日 provider 支持的地区：**

| provider | 适用范围 | 数据源 | 备注 |
|---|---|---|---|
| `china_timor`（默认） | 仅中国大陆 | timor.tech | 含调休补班日识别 |
| `nager` | 100+ 国家 | date.nager.at | 需填 `holiday_region`（ISO 国家码：US/JP/GB/DE/SG/MY/CA/AU 等） |

### 📡 API 接口

#### 任务（Task）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/task` | 提交新任务 |
| `GET` | `/task/{id}` | 查询任务状态和结果 |
| `GET` | `/tasks` | 列出所有任务 |
| `GET` | `/tasks/running` | **🆕** 列出当前正在执行的任务（含实时进度） |
| `POST` | `/task/{id}/cancel` | **🆕** 取消任务（pending 立即停，running 下轮中止） |
| `DELETE` | `/task/{id}` | 删除已完成/失败的任务 |
| `GET` | `/health` | 健康检查 |

#### 定时任务（Schedule）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/schedule` | 创建定时任务 |
| `GET` | `/schedules` | 列出所有定时任务（含下次触发时间） |
| `GET` | `/schedule/{id}` | 查询单个定时任务 |
| `PUT` | `/schedule/{id}` | 更新定时任务（启停/改 cron 等） |
| `DELETE` | `/schedule/{id}` | 删除定时任务 |
| `POST` | `/schedule/{id}/trigger` | **🆕** 立即手动触发一次（测试用） |

#### 任务状态

| 状态 | 说明 |
|------|------|
| `pending` | 排队等待设备连接 |
| `running` | 执行中 |
| `completed` | 成功完成 |
| `failed` | 失败（含被用户取消） |

### 📂 项目结构

```
AccessAgent/
├── main.py                    # 入口，启动 HTTP + WebSocket + Scheduler
├── config.py                  # 配置
├── mock_phone.py              # ADB 模拟手机
│
├── api/
│   └── app.py                 # FastAPI 路由（task + schedule）
│
├── core/
│   ├── server.py              # WebSocket + Agent 主循环（含取消检测）
│   ├── task_store.py          # 任务队列 + 取消逻辑
│   ├── scheduler.py           # 🆕 cron 调度器（APScheduler）
│   ├── schedule_store.py      # 🆕 定时任务持久化
│   ├── holiday.py             # 🆕 节假日检测（多 provider）
│   ├── ui_analyzer.py         # UI 树解析
│   └── annotator.py           # 截图标注
│
├── agents/
│   ├── planner.py             # 规划 Agent
│   ├── executor.py            # 执行 Agent
│   ├── reflector.py           # 反思 Agent
│   └── task_analyzer.py       # 前置分析
│
├── models/
│   └── llm.py                 # LLM 封装（含重试 / token 统计）
│
└── memory/
    ├── task_memory.py         # 任务记忆
    ├── step_fragment_memory.py# 步骤片段记忆
    └── experience_pool.py     # 经验池
```

### ⚙️ 配置参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `MAX_STEPS` | 50 | 单任务最大循环步数 |
| `MAX_RETRIES` | 2 | 连续失败多少次触发重新规划 |
| `MAX_TOTAL_FAILURES` | 12 | 累计失败上限 |
| `SCHEDULER_TIMEZONE` | Asia/Shanghai | 定时任务时区 |
| `CORS_ORIGINS` | localhost only | HTTP CORS 白名单 |
| `API_TOKEN` | 空 | 写操作鉴权 token |

### 🔒 注意事项

- `.env` 含 API Key，**绝不要提交到 Git**
- 定时打卡任务建议在 prompt 中明确「看到打卡时间才能 finish」，避免重试时重复打卡
- 节假日 API 失败时默认按工作日执行（避免错过打卡）
- `memory/*.json` 是运行时数据，已在 `.gitignore`

### 📜 License

MIT

---

## English

A multi-agent framework for Android phone automation. Combines Android Accessibility Service with vision AI to drive your phone with natural language.

### ✨ Features

- **Natural language tasks** — submit work via HTTP API, no scripting
- **Multi-agent collaboration** — Planner plans, Executor decides, Reflector verifies
- **Text-first + screenshots on demand** — uses the accessibility tree by default, only calls the vision model when needed (cost-efficient)
- **Dual-model architecture** — lightweight text LLM (DeepSeek) + multimodal LLM (Qwen-VL / GPT-4o)
- **Three-layer memory** — experience pool (general rules) + task memory (full-flow reuse) + step-fragment memory (cross-task substep reuse)
- **🆕 Scheduled tasks** — cron expressions + holiday skipping + retry on failure (perfect for daily clock-in)
- **🆕 Task cancellation / manual trigger / live progress** — full ops API
- **Token usage tracking** — per-step token + cost, with end-of-task summary

### 🏗️ Architecture

```
Caller
    │  POST /task / POST /schedule
    ▼
┌─────────────────────────────────────────┐
│         PC server (main.py)              │
│                                         │
│  FastAPI HTTP (:8000)                   │
│  ┌─────────────────────────────────┐   │
│  │ TaskStore  ←——  Scheduler (cron) │   │
│  └────────────────┬────────────────┘   │
│                   │                     │
│  WebSocket (:8765)│                     │
│  ┌────────────────▼────────────────┐   │
│  │     AccessAgentServer main loop   │   │
│  │  Planner → Executor → Reflector  │   │
│  └─────────────────────────────────┘   │
└────────────────────┬────────────────────┘
                     │ WebSocket
          ┌──────────▼──────────┐
          │   Android device     │
          │ Accessibility / ADB  │
          └─────────────────────┘
```

### 🚀 Quick start

#### 1. Clone & install

```bash
git clone https://github.com/boyovance/accessAgent.git
cd accessAgent
pip install -r requirements.txt
```

#### 2. Configure API keys

Copy `.env.example` to `.env` and fill in:

```ini
TEXT_API_KEY=your_deepseek_api_key
TEXT_BASE_URL=https://api.deepseek.com/v1
TEXT_MODEL=deepseek-chat

VISION_API_KEY=your_vision_api_key
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_MODEL=qwen-vl-plus

# Optional: HTTP security
CORS_ORIGINS=http://localhost:3000
API_TOKEN=your_random_token         # if set, write ops need X-API-Token header
SCHEDULER_TIMEZONE=Asia/Shanghai    # schedule timezone
```

#### 3. Start the server

```bash
python main.py
```

- HTTP API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- WebSocket: `ws://localhost:8765`

#### 4. Connect an Android device (mock_phone for dev)

Edit `DEVICE` and `ADB_PATH` at the top of `mock_phone.py`, then:

```bash
python mock_phone.py
```

#### 5. Submit a task

```bash
curl -X POST http://localhost:8000/task \
     -H "Content-Type: application/json" \
     -d '{"task": "Look up todays weather in Beijing"}'
```

### ⏰ Scheduled tasks (e.g. daily clock-in)

**Daily 9 AM weekday clock-in, skipping public holidays:**

```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Morning clock-in",
    "task": "Open the attendance app and clock in",
    "cron": "0 9 * * 1-5",
    "skip_holidays": true,
    "holiday_provider": "nager",
    "holiday_region": "US",
    "retry_max_attempts": 3,
    "retry_interval_seconds": 300
  }'
```

**Supported holiday providers:**

| Provider | Coverage | Source | Notes |
|---|---|---|---|
| `china_timor` (default) | Mainland China only | timor.tech | Recognizes makeup workdays |
| `nager` | 100+ countries | date.nager.at | Requires `holiday_region` (ISO code: US/JP/GB/DE/SG/MY/CA/AU…) |

### 📡 HTTP API

#### Tasks

| Method | Path | Description |
|------|------|------|
| `POST` | `/task` | Submit a new task |
| `GET` | `/task/{id}` | Query task status + result |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/running` | **🆕** List currently running tasks (with live progress) |
| `POST` | `/task/{id}/cancel` | **🆕** Cancel a task (pending stops instantly; running aborts next loop iter) |
| `DELETE` | `/task/{id}` | Delete completed/failed task record |
| `GET` | `/health` | Health check |

#### Schedules

| Method | Path | Description |
|------|------|------|
| `POST` | `/schedule` | Create a scheduled task |
| `GET` | `/schedules` | List all schedules (with next run time) |
| `GET` | `/schedule/{id}` | Get one schedule |
| `PUT` | `/schedule/{id}` | Update (enable/disable, change cron…) |
| `DELETE` | `/schedule/{id}` | Delete a schedule |
| `POST` | `/schedule/{id}/trigger` | **🆕** Manually trigger one execution (for testing) |

#### Task status

| Status | Description |
|------|------|
| `pending` | Queued, waiting for device |
| `running` | Executing |
| `completed` | Done |
| `failed` | Failed or cancelled by user |

### 📂 Project layout

```
AccessAgent/
├── main.py                    # entry: HTTP + WebSocket + Scheduler
├── config.py                  # configuration
├── mock_phone.py              # ADB-driven mock device
│
├── api/app.py                 # FastAPI routes (task + schedule)
│
├── core/
│   ├── server.py              # WebSocket + agent main loop (with cancel check)
│   ├── task_store.py          # task queue + cancellation
│   ├── scheduler.py           # 🆕 cron scheduler (APScheduler)
│   ├── schedule_store.py      # 🆕 schedule persistence
│   ├── holiday.py             # 🆕 holiday detection (pluggable providers)
│   ├── ui_analyzer.py         # accessibility tree parser
│   └── annotator.py           # screenshot annotation
│
├── agents/
│   ├── planner.py
│   ├── executor.py
│   ├── reflector.py
│   └── task_analyzer.py
│
├── models/llm.py              # LLM wrapper (retries + token stats)
│
└── memory/
    ├── task_memory.py
    ├── step_fragment_memory.py
    └── experience_pool.py
```

### ⚙️ Config knobs

| Setting | Default | Description |
|------|------|------|
| `MAX_STEPS` | 50 | Max loop iterations per task |
| `MAX_RETRIES` | 2 | Consecutive failures before replanning |
| `MAX_TOTAL_FAILURES` | 12 | Hard cap on total failures |
| `SCHEDULER_TIMEZONE` | Asia/Shanghai | Schedule timezone |
| `CORS_ORIGINS` | localhost only | HTTP CORS whitelist |
| `API_TOKEN` | empty | Token for write operations |

### 🔒 Notes

- `.env` contains API keys — **never commit it to git**
- For clock-in tasks, make the prompt explicit: "must see the clock-in timestamp before calling finish", to avoid double-clocking on retries
- If the holiday API fails, the scheduler defaults to "execute as workday" to avoid missed clock-ins
- All `memory/*.json` are runtime data, already in `.gitignore`

### 📜 License

MIT
