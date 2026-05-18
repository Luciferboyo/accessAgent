# AccessAgent

基于多 Agent 架构的 Android 手机自动化框架，结合无障碍服务（Accessibility Service）与视觉 AI，通过自然语言驱动手机完成复杂任务。

---

## 特性

- **自然语言下发任务**：通过 HTTP API 提交任务，无需编写脚本
- **多 Agent 协作**：Planner 规划、Executor 决策、Reflector 反思，三者分工协作
- **文字优先 + 按需截图**：优先使用无障碍树（AccessibilityInfo）做决策，仅在必要时调用视觉模型，节省成本
- **双模型架构**：文本决策用轻量 LLM（DeepSeek），视觉分析用多模态 LLM（Qwen-VL / GPT-4o）
- **任务记忆**：成功任务自动保存流程，相似任务直接复用，减少重复规划
- **任务队列**：支持多任务排队执行，HTTP API 实时查询状态
- **Token 用量统计**：每步打印 token 消耗与费用，任务结束汇总

---

## 架构

```
用户 / 调用方
    │  POST /task  {"task": "帮我搜索..."}
    ▼
┌─────────────────────────────────────────┐
│           PC 服务端 (main.py)            │
│                                         │
│  FastAPI HTTP (:8000)                   │
│  ┌─────────────────────────────────┐   │
│  │         任务队列 TaskStore        │   │
│  └────────────────┬────────────────┘   │
│                   │                     │
│  WebSocket (:8765)│                     │
│  ┌────────────────▼────────────────┐   │
│  │         AccessAgentServer        │   │
│  │  ┌──────────┐ ┌──────────────┐  │   │
│  │  │ Planner  │ │   Executor   │  │   │
│  │  │ 规划步骤  │ │ 文本/视觉决策 │  │   │
│  │  └──────────┘ └──────────────┘  │   │
│  │  ┌──────────┐ ┌──────────────┐  │   │
│  │  │Reflector │ │  TaskMemory  │  │   │
│  │  │ 验证结果  │ │   任务记忆   │  │   │
│  │  └──────────┘ └──────────────┘  │   │
│  └─────────────────────────────────┘   │
└────────────────────┬────────────────────┘
                     │ WebSocket
          ┌──────────▼──────────┐
          │    Android 设备      │
          │  无障碍 App / ADB    │
          │  读取 UI 树 + 执行   │
          └─────────────────────┘
```

### 单步执行流程

```
获取 UI 树
    │
    ▼
文本 LLM 决策（快速、低成本）
    │
    ├─ 信息充足 ──→ 直接执行动作
    │
    └─ 信息不足 ──→ 截图 + 视觉 LLM 决策
                          │
                          ▼
                     执行动作
                          │
                          ▼
                     Reflector 验证
                     （系统动作自动跳过）
                          │
                    成功 → 下一步
                    失败 → 重试 / 重新规划
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Luciferboyo/accessAgent.git
cd accessAgent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

```ini
# 文本模型（推荐 DeepSeek，速度快、成本低）
TEXT_API_KEY=your_deepseek_api_key
TEXT_BASE_URL=https://api.deepseek.com/v1
TEXT_MODEL=deepseek-chat

# 视觉模型（推荐 Qwen-VL-Plus 或 GPT-4o）
VISION_API_KEY=your_vision_api_key
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_MODEL=qwen-vl-plus
```

### 4. 启动服务

```bash
python main.py
```

服务启动后：
- HTTP API：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- WebSocket：`ws://localhost:8765`

### 5. 连接 Android 设备

**方式 A：使用模拟器（开发调试）**

确保已安装 ADB 并启动模拟器，修改 `mock_phone.py` 中的配置：

```python
DEVICE = "emulator-5554"   # adb devices 查看设备 ID
ADB_PATH = r"path/to/adb"
```

```bash
python mock_phone.py
```

**方式 B：使用真实 Android App**（开发中）

安装无障碍服务 App，在设置中启用无障碍权限，App 会自动连接 WebSocket 服务器。

### 6. 提交任务

```bash
# 提交任务
curl -X POST http://localhost:8000/task \
     -H "Content-Type: application/json" \
     -d '{"task": "帮我查询今天北京的天气"}'

# 返回
# {"task_id": "a1b2c3d4", "status": "pending", ...}

# 查询结果
curl http://localhost:8000/task/a1b2c3d4
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/task` | 提交新任务，返回 `task_id` |
| `GET` | `/task/{task_id}` | 查询任务状态和结果 |
| `GET` | `/tasks` | 列出所有任务（倒序） |
| `DELETE` | `/task/{task_id}` | 删除已完成/失败的任务记录 |
| `GET` | `/health` | 健康检查，返回队列状态 |

### 任务状态

| 状态 | 说明 |
|------|------|
| `pending` | 排队等待设备连接 |
| `running` | 执行中 |
| `completed` | 成功完成，`result` 字段含结果 |
| `failed` | 失败，`error` 字段含原因 |

---

## 支持的动作

| 动作 | 说明 |
|------|------|
| `click(index)` | 点击指定编号元素 |
| `long_click(index)` | 长按指定编号元素 |
| `type(index, text)` | 在输入框输入文字（支持中文，需 ADBKeyboard） |
| `scroll(index, direction)` | 滑动，方向：up/down/left/right |
| `back()` | 返回上一页 |
| `home()` | 回到桌面 |
| `open_app(package)` | 打开指定包名应用 |
| `search_web(query)` | 直接用 Chrome 搜索（绕过地址栏输入） |
| `finish()` | 操作类任务完成 |
| `report(content)` | 信息收集类任务汇报结果 |

---

## 项目结构

```
AccessAgent/
├── main.py                 # 入口，同时启动 HTTP + WebSocket 服务
├── config.py               # 配置（模型、价格、限制参数）
├── mock_phone.py           # ADB 模拟手机（开发调试用）
│
├── api/
│   └── app.py              # FastAPI HTTP 接口
│
├── core/
│   ├── server.py           # WebSocket 服务 + Agent 主循环
│   ├── task_store.py       # 任务队列与状态管理
│   ├── ui_analyzer.py      # 无障碍树解析与分析
│   └── annotator.py        # 截图标注（元素编号叠加）
│
├── agents/
│   ├── planner.py          # 任务规划 Agent
│   ├── executor.py         # 动作决策 Agent（文本 + 视觉）
│   └── reflector.py        # 结果验证 Agent
│
├── models/
│   └── llm.py              # LLM 封装（流式输出 + token 统计）
│
└── memory/
    └── task_memory.py      # 任务记忆（Jaccard 相似度复用）
```

---

## 配置参数

在 `config.py` 或 `.env` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_STEPS` | 30 | 单任务最大循环步数 |
| `MAX_RETRIES` | 3 | 连续失败多少次触发重新规划 |
| `MAX_TOTAL_FAILURES` | 10 | 累计失败上限，超过直接放弃 |
| `MAX_REPLANS` | 3 | 最多重新规划次数 |
| `TEXT_PRICE_INPUT` | 0.001 | 文本模型输入价格（元/千token） |
| `VISION_PRICE_INPUT` | 0.008 | 视觉模型输入价格（元/千token） |

---

## 中文输入支持

`adb shell input text` 不支持中文，输入中文需安装 ADBKeyboard：

1. 下载 [ADBKeyboard.apk](https://github.com/senzhk/ADBKeyBoard)
2. 安装并设置为默认输入法：
   ```bash
   adb install ADBKeyboard.apk
   adb shell ime set com.android.adbkeyboard/.AdbIME
   ```

---

## 注意事项

- `.env` 文件含有 API Key，**绝对不要提交到 Git**（已在 `.gitignore` 中排除）
- 视觉模型每次调用费用较高，系统默认连续失败 2 次才触发截图
- 任务记忆存储在 `memory/task_flows.json`，仅保存成功任务
- `mock_phone.py` 仅用于开发调试，生产环境请使用真实 Android App

---

## License

MIT
