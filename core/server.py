import asyncio
import base64
import functools
import json
import os
import websockets
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

from PIL import Image as _PILImage

from config import config
from utils import extract_json
from models.llm import TextLLM, VisionLLM, TokenUsage
from agents.planner import Planner
from agents.executor import Executor
from agents.reflector import Reflector
from agents.task_analyzer import TaskAnalyzer
from memory.task_memory import TaskMemory
from memory.experience_pool import ExperiencePool
from core.ui_analyzer import UIAnalyzer
from core.annotator import ScreenAnnotator
from core.task_store import TaskStore, TaskStatus


@dataclass
class UsageSummary:
    """统计整个任务的 token 用量和费用"""
    text_prompt: int = 0
    text_completion: int = 0
    text_cost: float = 0.0
    vision_prompt: int = 0
    vision_completion: int = 0
    vision_cost: float = 0.0

    def add_text(self, usage: TokenUsage):
        self.text_prompt += usage.prompt
        self.text_completion += usage.completion
        self.text_cost += usage.cost

    def add_vision(self, usage: TokenUsage):
        self.vision_prompt += usage.prompt
        self.vision_completion += usage.completion
        self.vision_cost += usage.cost

    @property
    def text_total(self): return self.text_prompt + self.text_completion

    @property
    def vision_total(self): return self.vision_prompt + self.vision_completion

    @property
    def grand_total(self): return self.text_total + self.vision_total

    @property
    def grand_cost(self): return self.text_cost + self.vision_cost

    def to_dict(self) -> dict:
        return {
            "text_prompt": self.text_prompt,
            "text_completion": self.text_completion,
            "text_cost": round(self.text_cost, 6),
            "vision_prompt": self.vision_prompt,
            "vision_completion": self.vision_completion,
            "vision_cost": round(self.vision_cost, 6),
            "grand_total": self.grand_total,
            "grand_cost": round(self.grand_cost, 6),
        }

    def print_final(self):
        print("\n" + "=" * 58)
        print("📊 Token 用量 & 费用汇总")
        print("=" * 58)
        print(f"  {'模型':<6} {'prompt':>8} {'completion':>10} {'tokens':>8} {'费用':>10}")
        print(f"  {'─'*52}")
        print(f"  {'文本':<6} {self.text_prompt:>8} {self.text_completion:>10} "
              f"{self.text_total:>8} {'¥'+f'{self.text_cost:.5f}':>10}")
        print(f"  {'视觉':<6} {self.vision_prompt:>8} {self.vision_completion:>10} "
              f"{self.vision_total:>8} {'¥'+f'{self.vision_cost:.5f}':>10}")
        print(f"  {'─'*52}")
        print(f"  {'合计':<6} {self.text_prompt+self.vision_prompt:>8} "
              f"{self.text_completion+self.vision_completion:>10} "
              f"{self.grand_total:>8} {'¥'+f'{self.grand_cost:.5f}':>10}")
        print("=" * 58)


class AccessAgentServer:
    def __init__(self, store: TaskStore):
        self.store = store

        text_llm = TextLLM(config.TEXT_API_KEY, config.TEXT_BASE_URL, config.TEXT_MODEL)
        vision_llm = VisionLLM(config.VISION_API_KEY, config.VISION_BASE_URL, config.VISION_MODEL)

        self.text_llm = text_llm
        self.vision_llm = vision_llm
        self.planner = Planner(text_llm)
        self.executor = Executor(text_llm, vision_llm)
        self.reflector = Reflector(text_llm)
        self.task_analyzer = TaskAnalyzer(text_llm)
        self.memory = TaskMemory()
        self.experience_pool = ExperiencePool()
        self.analyzer = UIAnalyzer()
        self.annotator = ScreenAnnotator()

        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    @staticmethod
    async def _in_thread(fn, *args, timeout: int = 150):
        """在线程池中执行同步阻塞函数，不阻塞事件循环，超时自动抛出"""
        loop = asyncio.get_running_loop()
        coro = loop.run_in_executor(None, functools.partial(fn, *args))
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"LLM 调用超时（>{timeout}s），请检查网络或 API 状态")

    async def _describe_screen(self, websocket, label: str = "init") -> tuple[str, TokenUsage]:
        """
        用 Vision 模型对当前屏幕做一句话定向描述。
        让 Planner 了解起始位置，避免规划冗余的导航步骤。
        label 用于区分保存文件名（"init" 表示任务开始，"replan_N" 表示第N次重规划前）。
        失败时静默降级，不影响主流程。
        """
        try:
            screenshot_b64 = await self._request_screenshot(websocket)
            # 保存截图（不标注，仅用于描述），统一保存为 JPEG 避免 PNG 体积过大
            img_path = os.path.join(config.SCREENSHOT_DIR, f"{label}.jpg")
            _raw = base64.b64decode(screenshot_b64)
            _PILImage.open(BytesIO(_raw)).convert("RGB").save(
                img_path, format="JPEG", quality=85, optimize=True
            )

            prompt = (
                "请用一句话（不超过80字）描述当前手机屏幕的状态。\n"
                "包含：① 所在的应用或页面类型  ② 页面的主要内容或当前 URL\n"
                "直接输出描述，不加任何前缀或解释。\n"
                "示例：Chrome浏览器显示 Google 搜索结果页，已搜索'骑士比赛'，顶部有比赛分数卡片。\n"
                "示例：Android 主屏幕，显示应用图标，当前无任何应用打开。\n"
                "示例：微信聊天列表页，显示多个联系人的最近消息。"
            )
            rsp, usage = await self._in_thread(self.vision_llm.predict, prompt, img_path)
            desc = rsp.strip()
            print(f"[定向] 当前页面：{desc}")
            return desc, usage
        except websockets.ConnectionClosed:
            # 连接断开不能静默吞掉，必须向上传播，让 handle() 正确标记任务失败
            raise
        except Exception as e:
            print(f"[定向] 页面描述失败（{e}），跳过定向截图")
            return "", TokenUsage()

    async def start(self, host: str, port: int):
        print(f"[WS] AccessAgent WebSocket 启动，监听 ws://{host}:{port}")
        async with websockets.serve(self.handle, host, port,
                                    max_size=10 * 1024 * 1024,
                                    ping_interval=None):
            await asyncio.Future()

    async def _classify_task(self, task: str) -> tuple[str, TokenUsage]:
        """
        用 LLM 判断任务类型，返回 (task_type, usage)：
        - "info"      信息收集类：目标是让用户获得某些信息，必须用 report 结束
        - "operation" 纯操作类：完成动作即可，用 finish 结束（打开应用、调整设置等）
        - "verify"    操作+验证类：完成操作后需确认结果成功，再用 finish 结束
                      （发消息需看到消息出现、点赞需看到已点赞、转账需看到成功提示）
        只调用一次，结果缓存在 handle() 里。
        """
        prompt = f"""判断下面这个手机自动化任务的类型：

任务：{task}

三种类型定义：
- info（信息收集）：用户最终目的是"获得某些信息"（查询数据、搜索内容、读取结果、告知用户某个值）
- operation（纯操作）：完成动作即可，不需要确认结果（打开应用、调整系统设置、拍照、截图）
- verify（操作+验证）：需要完成某个操作，且必须在界面上看到成功标志才算完成
  （发送消息需看到消息显示在对话框、点赞需看到按钮变色、转账需看到成功页面、预订需看到确认单）

判断优先级：
1. 含"告诉我"、"查询"、"搜索"、"汇报"→ info
2. 含"发送"、"提交"、"转账"、"预订"、"点赞"、"关注"、"评论"等需要确认结果的 → verify
3. 含"打开"、"设置"、"调整"、"拍照"等不需要确认的 → operation

只输出 JSON：{{"task_type": "info/operation/verify", "reason": "一句话说明判断依据"}}"""

        classify_usage = TokenUsage()
        try:
            rsp, classify_usage = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            task_type = data.get("task_type", "operation")
            if task_type not in ("info", "operation", "verify"):
                task_type = "operation"
            label = {"info": "信息收集类", "operation": "纯操作类", "verify": "操作+验证类"}
            print(f"[任务分类] {label[task_type]} | {data.get('reason', '')}")
            return task_type, classify_usage
        except Exception as e:
            print(f"[任务分类] 解析失败（{e}），默认为 operation")
            return "operation", classify_usage

    async def handle(self, websocket):
        print("[WS] Android App 已连接，等待任务...")

        # 从队列取下一个待执行任务
        task_id = await self.store.queue.get()
        record = self.store.get(task_id)
        if record is None:
            print(f"[错误] 任务 {task_id} 在 store 中不存在，跳过")
            return
        task = record.task
        # 优先使用用户提交时指定的 max_steps，否则使用全局配置
        max_steps = record.max_steps if record.max_steps and record.max_steps > 0 else config.MAX_STEPS

        print(f"[WS] 开始执行任务 [{task_id}]：{task}（最大步数：{max_steps}）")
        self.store.update(task_id, status=TaskStatus.RUNNING)

        await websocket.send(json.dumps({"type": "task", "task": task}))

        # 从记忆中查找相似任务
        # full  → 直接复用计划，planner_hint = None
        # partial → 不复用计划，将 hint 传给 Planner 避免重复弯路
        memory_result = self.memory.find_similar(task)
        if memory_result is None:
            plan = None
            planner_hint = None
        elif memory_result["quality"] == "full":
            plan = memory_result["steps"] or None   # 空列表视同未找到
            planner_hint = None
        else:                                        # partial
            plan = None
            planner_hint = memory_result.get("hint")
            print("[Memory] 上次部分成功，经验将作为规划参考")

        step_index = 0
        history = []
        action_log = []
        consecutive_failures = 0
        total_failures = 0
        replan_count = 0
        failure_reason = ""
        usage = UsageSummary()
        final_result = None
        success = False
        report_rejected_count = 0  # report 被校验拒绝的次数，超限后强制放行
        verify_failed_count = 0    # verify 任务 finish 未通过次数，超限后强制放行
        force_accepted = False     # 是否为强制放行（用于记忆质量判断）
        force_vision_next_step = False  # tap 之后强制截图，避免 Text 用错坐标重点
        stuck_action_counts: dict[str, int] = {}  # 记录每个 (action:index) 累计卡住次数
        last_tap_coords: tuple[int, int] | None = None   # 上一次 tap 的手机坐标
        consecutive_same_tap = 0                         # 连续在相同位置 tap 的次数

        try:
            # 任务开始时分类一次，后续复用，不重复调用 LLM
            # task_type: "info" | "operation" | "verify"
            # 放在 try 内：若分类期间连接断开，finally 可正确标记任务失败
            task_type, classify_usage = await self._classify_task(task)
            usage.add_text(classify_usage)

            # ── 前置分析：前提验证 + 知识预判 ───────────────────────
            today_str = datetime.now().strftime("%Y-%m-%d")
            print("[TaskAnalyzer] 分析任务前提与知识预判...")
            analysis, analysis_usage = await self._in_thread(
                self.task_analyzer.analyze, task, task_type, today_str
            )
            usage.add_text(analysis_usage)

            if not analysis.get("valid", True):
                # 任务前提不合理，直接拒绝，告知用户原因
                issue = analysis.get("issue", "任务前提存在问题")
                print(f"[TaskAnalyzer] ⚠ 任务前提有误：{issue}")
                await websocket.send(json.dumps({
                    "type": "finish",
                    "message": f"任务无法执行：{issue}"
                }))
                self.store.update(task_id,
                                  status=TaskStatus.FAILED,
                                  error=f"任务前提有误：{issue}",
                                  completed_at=datetime.now().isoformat(),
                                  usage=usage.to_dict())
                return  # finally 仍会执行 print_final

            if task_type == "info" and analysis.get("can_answer_directly"):
                # LLM 可以直接回答，无需手机操作
                direct_answer = analysis.get("direct_answer", "")
                if direct_answer:
                    print("[TaskAnalyzer] ✓ 从 AI 知识库直接回答，无需手机操作")
                    answer_with_note = (
                        f"[此答案来自 AI 知识库，无需实时搜索]\n\n{direct_answer}"
                    )
                    self._save_report(task, answer_with_note)
                    await websocket.send(json.dumps({
                        "type": "finish",
                        "message": "信息收集完成（AI 直接回答）"
                    }))
                    final_result = direct_answer
                    success = True
                    return  # finally 仍会执行，并根据 success=True 写入 COMPLETED

            # 将 AI 预判信息注入 planner_hint，让 Planner 生成更有针对性的计划
            hypothesis = analysis.get("hypothesis", "")
            search_hint = analysis.get("search_hint", "")
            if hypothesis or search_hint:
                if planner_hint is None:
                    planner_hint = {}
                if hypothesis:
                    planner_hint["hypothesis"] = hypothesis
                if search_hint:
                    planner_hint["search_hint"] = search_hint
                print(f"[TaskAnalyzer] 知识预判已注入 Planner："
                      f"{'hypothesis=' + hypothesis[:60] if hypothesis else ''}"
                      f"{'  search_hint=' + search_hint[:60] if search_hint else ''}")

            # ────────────────────────────────────────────────────────
            current_state = await self._request_state(websocket)
            consecutive_scroll_no_change = 0   # 连续 scroll 无变化次数

            for global_step in range(max_steps):
                print(f"\n{'='*50}")
                print(f"[Step {global_step + 1}] [{task_id}]")

                # ── 1. 使用当前状态 ──────────────────────────────
                ui_elements = current_state.get("ui_elements", [])
                ui_text = self.analyzer.parse_elements(ui_elements)
                print(f"[AccessInfo] {len(ui_elements)} 个可交互元素")

                # ── 2. 首轮生成计划 ──────────────────────────────
                if plan is None:
                    # 定向截图：先用 Vision 描述当前页面，帮助 Planner 了解起始位置
                    print("[定向] 截图分析当前页面状态...")
                    screen_desc, desc_usage = await self._describe_screen(websocket)
                    usage.add_vision(desc_usage)

                    print("[Planner] 生成任务计划...")
                    _plan_app_pkg = current_state.get("package", "")
                    _plan_exp = self.experience_pool.format_for_prompt(
                        self.experience_pool.search(task=task, app_package=_plan_app_pkg)
                    )
                    if _plan_exp:
                        print(f"[Experience] Planner 命中经验，已注入")
                    plan, plan_usage = await self._in_thread(
                        self.planner.make_plan, task, ui_text, planner_hint, screen_desc,
                        _plan_exp
                    )
                    usage.add_text(plan_usage)
                    # 生成失败时保底：把整个任务作为一个步骤
                    if not plan:
                        plan = [task]
                    print(f"[Planner] 计划步骤：{plan}")
                    print(f"[Token] 规划用量：{plan_usage}")

                if plan is None or step_index >= len(plan):
                    if task_type == "info":
                        # 信息收集类任务走完所有计划步骤，但没有 report，强制补一步
                        # 防止重复追加：检查最后一步是否已经是汇报步骤
                        report_step = "将目前收集到的所有信息整理后用 report 汇报给用户，如果信息不完整请说明原因"
                        if plan and plan[-1] == report_step:
                            # 汇报步骤已经存在但仍未执行成功，说明执行器无法完成
                            # 强制结束，避免无限追加
                            print("[注意] 汇报步骤已追加但仍未完成，强制以当前状态结束任务")
                            await websocket.send(json.dumps({"type": "finish", "message": "信息收集未能完成，任务结束"}))
                            self.store.update(task_id,
                                              status=TaskStatus.FAILED,
                                              error="信息收集步骤多次无法完成",
                                              completed_at=datetime.now().isoformat(),
                                              usage=usage.to_dict())
                            break
                        print("[注意] 信息收集任务步骤已完成，但尚未汇报结果，追加汇报步骤")
                        if plan is None:
                            plan = []
                        plan.append(report_step)
                        failure_reason = "所有规划步骤已完成，现在必须用 report 汇报收集到的内容"
                        continue
                    # verify 任务需要在所有步骤完成后再做一次终态确认
                    if task_type == "verify":
                        print("[完成] 所有步骤执行完毕，执行操作结果终态验证...")
                        verified, verify_reason, verify_op_usage = await self._verify_operation(
                            task, ui_text, websocket, len(ui_elements)
                        )
                        usage.add_text(verify_op_usage)
                        if not verified:
                            print(f"[验证] 终态验证未通过：{verify_reason}，继续尝试")
                            consecutive_failures += 1
                            total_failures += 1
                            failure_reason = verify_reason or "界面未显示操作成功的明确信号"
                            continue
                        print(f"[验证] 终态验证通过：{verify_reason}")
                    print("[完成] 所有步骤执行完毕")
                    await websocket.send(json.dumps({"type": "finish", "message": "任务完成"}))
                    self.memory.save_flow(task, plan, action_log, quality="full")
                    success = True
                    break

                current_step = plan[step_index]
                print(f"[步骤 {step_index + 1}/{len(plan)}] {current_step}")

                # ── 实时进度更新（供外部轮询 /task/{id} 查看）──────
                self.store.update(
                    task_id,
                    progress=f"步骤 {step_index + 1}/{len(plan)}：{current_step[:60]}",
                    current_step=step_index + 1,
                    total_steps=len(plan),
                )

                # ── 2.5 检索相关操作经验，注入当前步骤提示 ──────────
                _app_pkg = current_state.get("package", "")
                _exp_hits = self.experience_pool.search(
                    task=task, app_package=_app_pkg, history=history
                )
                _experiences_text = self.experience_pool.format_for_prompt(_exp_hits)
                if _exp_hits:
                    print(f"[Experience] 命中 {len(_exp_hits)} 条经验：{[e['id'] for e in _exp_hits]}")

                # ── 3. 优先文本决策 ──────────────────────────────
                print("[Text] 尝试文本决策...")
                action, text_usage = await self._in_thread(
                    self.executor.decide_text, current_step, step_index, len(plan),
                    ui_text, history, failure_reason, consecutive_failures, task_type,
                    _experiences_text
                )
                usage.add_text(text_usage)
                print(f"[Text] {action.get('action')} | {action.get('reason')}")
                print(f"[Token] 文本决策：{text_usage}")

                # ── 4. 判断是否需要截图 ──────────────────────────
                # 文本模型已决定 report/finish/find_package 时，不触发 Vision（避免被覆盖）
                text_action = action.get("action")
                if text_action in ("report", "finish", "find_package", "tap"):
                    need_vision = False
                    _force_by_tap = False
                else:
                    _force_by_tap = force_vision_next_step
                    need_vision = (
                        text_action == "need_screenshot"
                        or self.analyzer.needs_screenshot(ui_elements)
                        or consecutive_failures >= 2
                        or _force_by_tap  # 上一步为 tap，强制截图确认结果
                    )
                    if _force_by_tap:
                        print("[Vision] 上一步为 tap 操作，强制截图确认结果")
                force_vision_next_step = False  # 消费后重置

                vision_usage = None
                if need_vision:
                    if action.get("action") == "need_screenshot":
                        trigger = "文本信息不足"
                    elif consecutive_failures >= 2:
                        trigger = f"连续失败 {consecutive_failures} 次（{failure_reason}）"
                    elif _force_by_tap:
                        trigger = "上一步 tap 操作，确认结果"
                    else:
                        trigger = "界面元素无有效文字"

                    print(f"[Vision] 触发截图，原因：{trigger}")
                    screenshot_b64 = await self._request_screenshot(websocket)
                    # annotator 做文件 IO + PIL 处理，放线程池避免阻塞事件循环
                    img_path = await self._in_thread(
                        self.annotator.save_screenshot,
                        screenshot_b64, config.SCREENSHOT_DIR, global_step
                    )
                    screen_size = current_state.get("screen_size", [1080, 2340])
                    # 标注路径：去掉原扩展名，加 _labeled.jpg（annotate 内部保存为 JPEG）
                    stem = os.path.splitext(img_path)[0]
                    annotated = await self._in_thread(
                        self.annotator.annotate,
                        img_path, ui_elements,
                        stem + "_labeled.jpg",
                        screen_size
                    )
                    print(f"[Vision] 标注截图：{annotated}")
                    # 计算截图尺寸（mock_phone 压缩至 max_width=720）
                    _scale = 720 / screen_size[0] if screen_size[0] > 0 else 1.0
                    img_size = (720, int(screen_size[1] * _scale))
                    action, vision_usage = await self._in_thread(
                        self.executor.decide_vision,
                        current_step, step_index, len(plan), annotated, ui_text, history,
                        failure_reason, screen_size, img_size, consecutive_failures, task_type,
                        _experiences_text
                    )
                    usage.add_vision(vision_usage)
                    print(f"[Vision] {action.get('action')} | {action.get('reason')}")
                    print(f"[Token] 视觉决策：{vision_usage}")

                # ── 5. 任务完成或汇报结果 ────────────────────────
                if action.get("action") == "finish":
                    if task_type == "info":
                        print("[拦截] 信息收集任务不允许 finish，继续执行...")
                        failure_reason = "任务需要收集信息并用 report 汇报结果，绝对不能用 finish 结束"
                        consecutive_failures += 1
                        current_state = await self._request_state(websocket)
                        continue

                    if task_type == "verify":
                        # 操作+验证类：finish 前先确认操作结果
                        MAX_VERIFY_FAILURES = 3
                        if verify_failed_count >= MAX_VERIFY_FAILURES:
                            print(f"[放行] 操作验证已失败 {verify_failed_count} 次，强制接受 finish")
                            verified, verify_reason = True, "多次验证后强制放行"
                            force_accepted = True
                        else:
                            print("[验证] 检查操作是否成功...")
                            verified, verify_reason, verify_op_usage = await self._verify_operation(
                                task, ui_text, websocket, len(ui_elements)
                            )
                            usage.add_text(verify_op_usage)

                        if not verified:
                            verify_failed_count += 1
                            remaining = MAX_VERIFY_FAILURES - verify_failed_count
                            print(f"[拦截] 操作结果未确认（第 {verify_failed_count} 次）：{verify_reason}")
                            if remaining > 0:
                                failure_reason = (
                                    f"操作尚未确认成功：{verify_reason}。"
                                    f"请继续操作直到界面出现明确的成功标志。"
                                    f"（还有 {remaining} 次机会）"
                                )
                            else:
                                failure_reason = (
                                    f"操作结果多次无法确认（{verify_reason}）。"
                                    f"这是最后机会：请截图确认当前界面状态后再决定是否 finish。"
                                )
                            consecutive_failures += 1
                            current_state = await self._request_state(websocket)
                            continue
                        print(f"[验证] 操作已确认成功：{verify_reason}")

                    await websocket.send(json.dumps({"type": "finish", "message": "任务完成"}))
                    if force_accepted:
                        hint = self._build_partial_hint(action_log, None)
                        self.memory.save_flow(task, plan, action_log, quality="partial", hint=hint)
                    else:
                        self.memory.save_flow(task, plan, action_log, quality="full")
                    success = True
                    print("[完成] 任务成功结束")
                    break

                if action.get("action") == "report":
                    # operation / verify 类任务不应以 report 结束，引导改为 finish
                    # verify 任务必须在界面上看到成功标志后才能 finish，
                    # 用 report 说明"任务失败"会被误判为成功并污染记忆库
                    if task_type in ("operation", "verify"):
                        type_label = "纯操作" if task_type == "operation" else "操作+验证"
                        print(f"[拦截] {type_label}任务不应使用 report，引导改用 finish")
                        failure_reason = (
                            f"这是一个{type_label}类任务，目标是完成操作"
                            + ("" if task_type == "operation" else "并在界面上确认成功")
                            + "，而非收集信息。"
                            "请不要使用 report。"
                            + ("确认操作已完成后直接调用 finish。"
                               if task_type == "operation"
                               else "继续操作直到界面出现成功标志，然后调用 finish。")
                        )
                        # 直接拉到 2，确保下一步立即触发 🚨 强警告，不被 progress 衰减稀释
                        consecutive_failures = max(consecutive_failures + 1, 2)
                        current_state = await self._request_state(websocket)
                        continue

                    content = action.get("params", {}).get("content", "")

                    # report 被拒次数超过阈值：强制放行，接受最优努力结果
                    MAX_REPORT_REJECTIONS = 3
                    if report_rejected_count >= MAX_REPORT_REJECTIONS:
                        print(f"[放行] report 已被拒绝 {report_rejected_count} 次，强制接受当前内容")
                        valid, missing = True, ""
                        force_accepted = True
                    else:
                        # 校验 report 内容是否真正回答了任务
                        valid, missing, validate_usage = await self._validate_report(task, content)
                        usage.add_text(validate_usage)

                    if not valid:
                        report_rejected_count += 1
                        remaining = MAX_REPORT_REJECTIONS - report_rejected_count
                        print(f"[拦截] report 内容不符合任务要求（第 {report_rejected_count} 次拒绝）：{missing}")
                        if remaining > 0:
                            failure_reason = (
                                f"report 的内容不完整或不相关：{missing}。"
                                f"请继续操作，获取更完整的数据后再 report。"
                                f"（还有 {remaining} 次机会，之后将强制接受）"
                            )
                        else:
                            failure_reason = (
                                f"report 内容多次被拒绝（{missing}）。"
                                f"这是最后一次机会：请将目前找到的所有信息汇总，"
                                f"并明确说明哪些信息无法获取及原因，然后用 report 汇报。"
                            )
                        consecutive_failures += 1
                        current_state = await self._request_state(websocket)
                        continue

                    self._save_report(task, content)
                    await websocket.send(json.dumps({"type": "finish", "message": "信息收集完成"}))
                    if force_accepted:
                        hint = self._build_partial_hint(action_log, content)
                        self.memory.save_flow(task, plan, action_log, quality="partial", hint=hint)
                    else:
                        self.memory.save_flow(task, plan, action_log, quality="full")
                    final_result = content
                    success = True
                    print("[完成] 信息收集完成")
                    break

                # ── 6. 执行动作 ──────────────────────────────────
                cmd = self._build_command(action, ui_elements,
                                          current_state.get("screen_size"))
                if cmd is None:
                    # index 越界或元素 bounds 无效（不可见），跳过指令直接重试
                    print("[警告] 元素不存在或不可见，跳过本次指令，重新获取界面状态")
                    # 元素极少（≤3）说明被困在弹窗/单按钮页面，Vision 在幻觉不存在的关闭按钮
                    # 自动执行 back() 兜底脱困，避免死在弹窗里
                    if len(ui_elements) <= 3:
                        print("[兜底] 当前仅有极少元素，自动执行 back() 尝试脱困")
                        await websocket.send(json.dumps({"type": "back"}))
                        await websocket.recv()  # 消费执行结果
                    consecutive_failures += 1
                    total_failures += 1
                    failure_reason = (
                        f"AI 指定的元素编号不存在或该元素不可见（零尺寸）。"
                        f"当前共 {len(ui_elements)} 个元素，请只选择截图或文字描述中实际可见的元素编号。"
                        + ("已自动执行 back() 尝试退出当前页面。" if len(ui_elements) <= 3 else "")
                    )
                    current_state = await self._request_state(websocket)
                    continue

                print(f"[执行] 发送指令：{cmd}")
                await websocket.send(json.dumps(cmd))
                action_log.append(action)

                result_raw = await websocket.recv()
                try:
                    result = json.loads(result_raw)
                except json.JSONDecodeError:
                    print(f"[警告] 动作执行响应解析失败，原始内容：{str(result_raw)[:100]}")
                    result = {"status": "unknown"}

                # ── find_package 特殊处理：注入查询结果，重新让 Executor 决策 ──
                if action.get("action") == "find_package":
                    packages = result.get("packages", [])
                    keyword = action.get("params", {}).get("keyword", "")
                    if packages:
                        pkg_str = "、".join(packages)
                        failure_reason = (
                            f"find_package 已查到设备上与 '{keyword}' 匹配的包名：{pkg_str}。"
                            f"请直接使用正确的包名调用 open_app。"
                        )
                        history.append(
                            f"步骤{step_index + 1}：find_package({keyword}) → {pkg_str}"
                        )
                        print(f"[find_package] 找到包名：{pkg_str}")
                        consecutive_failures = 0  # 包名查询成功，不算失败
                    else:
                        failure_reason = (
                            f"find_package 未找到关键词 '{keyword}' 对应的已安装应用，"
                            f"请确认应用是否已安装，或尝试换一个更短的关键词重新查询。"
                        )
                        history.append(
                            f"步骤{step_index + 1}：find_package({keyword}) → 未找到匹配包名"
                        )
                        print(f"[find_package] 未找到匹配包名")
                        consecutive_failures += 1
                        total_failures += 1
                    current_state = await self._request_state(websocket)
                    continue  # 不走 Reflector，直接让 Executor 用包名信息重新决策

                print(f"[App] 执行状态：{result.get('status')}")

                # ── 7. 获取新状态（复用为下一步）────────────────
                ui_text_before = ui_text
                current_state = await self._request_state(websocket)
                ui_text_after = self.analyzer.parse_elements(
                    current_state.get("ui_elements", [])
                )

                # ── 8. 自我反思（系统级动作直接跳过，节省 token）────
                # tap 专用于 WebView/小程序内容，无障碍树不可见任何变化，Reflector
                # 必然误报失败——跳过 Reflector，由下一步的截图/状态判断是否成功
                SKIP_REFLECT = {"back", "home", "search_web", "tap"}
                act_name = action.get("action")
                if act_name == "open_app":
                    # open_app 用包名验证：确认前台应用确实切换为目标应用
                    target_pkg = action.get("params", {}).get("package", "")
                    actual_pkg = current_state.get("package", "")
                    if target_pkg and actual_pkg and actual_pkg != target_pkg:
                        verify = {
                            "status": "stuck",
                            "reason": f"open_app 后前台应用是 {actual_pkg}，而非目标 {target_pkg}，应用未能成功切换",
                            "next_hint": ""
                        }
                        print(f"[open_app] 切换失败：目标={target_pkg}，实际={actual_pkg}")
                    else:
                        # 仅当当前步骤的核心目标是"打开应用"时，才将本步标记为 done
                        # 若步骤目标是应用内的导航/操作（如找图片、转发等），
                        # open_app 只是纠偏动作，标记为 progress，避免错误跳过剩余步骤目标
                        _step_is_open = (
                            "open_app" in current_step
                            or current_step.strip().startswith(("打开", "启动", "launch", "open", "Open"))
                        )
                        _status = "done" if _step_is_open else "progress"
                        _hint = "" if _step_is_open else (
                            f"已成功打开 {actual_pkg or target_pkg}，请继续在应用内完成当前步骤的目标"
                        )
                        verify = {
                            "status": _status,
                            "reason": f"open_app 后前台应用已切换为 {actual_pkg or target_pkg}",
                            "next_hint": _hint
                        }
                        print(f"[open_app] 切换成功：{actual_pkg or target_pkg}（步骤状态：{_status}）")
                    reflect_usage = TokenUsage()
                elif act_name in SKIP_REFLECT:
                    verify = {
                        "status": "progress",
                        "reason": f"{act_name} 已执行（系统动作无需反思，继续推进步骤）",
                        "next_hint": ""
                    }
                    reflect_usage = TokenUsage()
                    if act_name == "tap":
                        # tap 专用于 WebView 内无编号按钮，结果无法从无障碍树读取
                        # 强制下一步截图确认，避免 Text 模型用错误坐标重复操作
                        force_vision_next_step = True

                        # ── 连续相同位置 tap 检测 ────────────────────────
                        # 若多次在同一坐标（±60px）tap 无变化，说明：
                        # 联系人已被选中（再 tap 会取消），或位置不对，需提示换策略
                        tap_x = action.get("params", {}).get("x", 0)
                        tap_y = action.get("params", {}).get("y", 0)
                        if last_tap_coords is not None:
                            dx = abs(tap_x - last_tap_coords[0])
                            dy = abs(tap_y - last_tap_coords[1])
                            if dx <= 60 and dy <= 60:
                                consecutive_same_tap += 1
                            else:
                                consecutive_same_tap = 0   # 位置改变，重置
                        last_tap_coords = (tap_x, tap_y)

                        if consecutive_same_tap >= 2:
                            # 连续 3 次（0,1,2）相同位置 tap → 注入提示，触发视觉重新审视
                            _tap_warn = (
                                f"⚠️ 已连续 {consecutive_same_tap + 1} 次在相近坐标 "
                                f"({tap_x},{tap_y}) 执行 tap，该位置可能在反复切换选中/取消选中状态。\n"
                                f"【立即执行以下三步检查】：\n"
                                f"① 查看截图【底部】：若有蓝色 [Share in N chat(s)] 或 [发送给 N 人] 按钮 → "
                                f"说明联系人已选中，必须 tap 该底部按钮（不要再 tap 联系人），操作完成！\n"
                                f"② 若底部无发送按钮，查看联系人头像左侧是否有勾选圆圈/蓝点 → "
                                f"若有，等待底部按钮出现或尝试 tap 头像圆圈（x 约 {max(30, tap_x - 60)}，y 约 {tap_y}）。\n"
                                f"③ 若联系人未被选中（无勾选标记），尝试点击更靠右的区域（联系人名称文字中心，x 坐标约 +80 到 +150）。"
                            )
                            failure_reason = _tap_warn
                            consecutive_failures += 1
                            total_failures += 1
                            print(f"[tap检测] 连续相同位置 {consecutive_same_tap + 1} 次，注入提示")
                    else:
                        # 非 tap 系统动作：重置同位置 tap 计数
                        consecutive_same_tap = 0
                        last_tap_coords = None
                    print(f"[Reflector] 跳过（{act_name} 系统动作），标记为 progress")
                else:
                    print("[Reflector] 自我反思...")
                    verify, reflect_usage = await self._in_thread(
                        self.reflector.verify, current_step, action,
                        ui_text_before, ui_text_after, _experiences_text
                    )
                    usage.add_text(reflect_usage)
                    reflect_status_dbg = verify.get("status", "?")
                    print(f"[Reflector] {reflect_status_dbg} | {verify.get('reason')}")
                    print(f"[Token] 反思用量：{reflect_usage}")

                # ── 9. 本步 token 小计 ───────────────────────────
                step_total = text_usage.total + reflect_usage.total
                if need_vision and vision_usage:
                    step_total += vision_usage.total
                print(f"[Token] 本步合计：{step_total} | 累计：{usage.grand_total}")

                # ── 10. 三状态处理：done / progress / stuck ──────
                reflect_status = verify.get("status")
                if reflect_status is None:
                    # 兼容旧格式 success: true/false
                    reflect_status = "done" if verify.get("success") else "stuck"

                # scroll 连续无变化检测：到达边界后继续 scroll 无意义，强制转 stuck
                if (act_name == "scroll"
                        and reflect_status == "progress"
                        and "无变化" in verify.get("reason", "")):
                    consecutive_scroll_no_change += 1
                    if consecutive_scroll_no_change >= 2:
                        reflect_status = "stuck"
                        verify["status"] = "stuck"
                        verify["reason"] = (
                            f"连续 {consecutive_scroll_no_change} 次 scroll 界面无变化，"
                            "列表已到边界，继续 scroll 无效"
                        )
                        verify["next_hint"] = (
                            "列表已到边界，请直接对当前可见内容执行操作"
                            "（如长按图片消息、点击目标元素），不要继续 scroll"
                        )
                        print(f"[Scroll] 连续 {consecutive_scroll_no_change} 次无变化，强制转 stuck")
                else:
                    consecutive_scroll_no_change = 0

                status_label = {
                    "done": "完成", "progress": "进行中", "stuck": "失败"
                }.get(reflect_status, "失败")
                history.append(
                    f"步骤{step_index + 1}：{current_step[:50]} -> "
                    f"{action.get('action')} -> [{status_label}] {verify.get('reason', '')[:60]}"
                )

                if reflect_status == "done":
                    step_index += 1
                    consecutive_failures = 0
                    failure_reason = ""
                    consecutive_same_tap = 0   # 步骤完成，重置 tap 计数
                    last_tap_coords = None
                elif reflect_status == "progress":
                    # 有效推进，不计为失败，不换步骤
                    # 用衰减而非全量重置：避免 stuck/progress 交替时永远触不发重规划
                    consecutive_failures = max(0, consecutive_failures - 1)
                    failure_reason = ""
                    next_hint = verify.get("next_hint", "")
                    if next_hint:
                        history.append(f"  → 提示：{next_hint}")
                    print(f"[进行中] 步骤推进中：{verify.get('reason', '')}")
                elif reflect_status == "stuck":
                    consecutive_failures += 1
                    total_failures += 1
                    failure_reason = verify.get("reason", "")
                    # stuck 时也把 next_hint（替代方案）加入历史，让下一步 Executor 看到
                    next_hint_stuck = verify.get("next_hint", "")
                    if next_hint_stuck:
                        history.append(f"  → 建议改用：{next_hint_stuck}")
                        # failure_reason 追加替代建议，传给 Executor prompt
                        failure_reason = f"{failure_reason}；建议改用：{next_hint_stuck}"

                    # 跨步骤卡死检测：同一 (action, index) 组合累计卡死 2 次以上
                    # 即使中间有 progress（如 back），也说明该路径根本不可行
                    stuck_key = f"{act_name}:{action.get('params', {}).get('index', '')}"
                    stuck_action_counts[stuck_key] = stuck_action_counts.get(stuck_key, 0) + 1
                    if stuck_action_counts[stuck_key] >= 2:
                        failure_reason = (
                            f"⚠️ 操作 [{stuck_key}] 已累计卡死 {stuck_action_counts[stuck_key]} 次"
                            f"（即使中间有返回操作也无法解决），该坐标/元素路径完全不可行！"
                            f"原因：{failure_reason}。"
                            f"必须改用截图重新识别正确目标，或换用完全不同的操作路径。"
                        )
                        # 强制下一步截图，让 Vision 重新识别正确目标
                        force_vision_next_step = True
                        print(f"[卡死检测] {stuck_key} 累计卡死 {stuck_action_counts[stuck_key]} 次，强制下步截图")

                    print(f"[失败] 累计连续失败 {consecutive_failures} 次（总计 {total_failures} 次）")

                    if total_failures >= config.MAX_TOTAL_FAILURES:
                        error_msg = f"累计失败{total_failures}次，{failure_reason}"
                        await websocket.send(json.dumps({
                            "type": "finish",
                            "message": f"任务失败：{error_msg}"
                        }))
                        self.store.update(task_id,
                                          status=TaskStatus.FAILED,
                                          error=error_msg,
                                          completed_at=datetime.now().isoformat(),
                                          usage=usage.to_dict())
                        break

                    if consecutive_failures >= config.MAX_RETRIES:
                        replan_count += 1
                        if replan_count > config.MAX_REPLANS:
                            error_msg = f"重新规划{replan_count}次后仍无法完成"
                            await websocket.send(json.dumps({
                                "type": "finish",
                                "message": f"任务失败：{error_msg}"
                            }))
                            self.store.update(task_id,
                                              status=TaskStatus.FAILED,
                                              error=error_msg,
                                              completed_at=datetime.now().isoformat(),
                                              usage=usage.to_dict())
                            break
                        print(f"[Planner] 连续失败，第 {replan_count} 次重新规划...")
                        self.store.update(
                            task_id,
                            progress=f"重新规划中（第 {replan_count} 次），原因：{failure_reason[:50]}",
                        )

                        # 重规划前：拍截图了解当前页面位置，避免规划出已在当前位置之前的步骤
                        print("[重规划] 截图分析当前页面状态...")
                        screen_desc_replan, desc_usage2 = await self._describe_screen(
                            websocket, label=f"replan_{replan_count}"
                        )
                        usage.add_vision(desc_usage2)

                        # 从 history 中提取已尝试过的关键路径，告知 Planner 避免重复
                        tried_approaches = self._summarize_tried_approaches(history)

                        plan, replan_usage = await self._in_thread(
                            self.planner.revise_plan,
                            task, plan, step_index, failure_reason, ui_text_after,
                            tried_approaches, screen_desc_replan, _experiences_text
                        )
                        usage.add_text(replan_usage)
                        print(f"[Token] 重规划用量：{replan_usage}")
                        if not plan:
                            plan = [task]   # LLM 返回空列表时保底，避免假成功
                        step_index = 0        # 新计划从第 0 步开始
                        consecutive_failures = 0
                        failure_reason = ""
                        consecutive_same_tap = 0   # 重规划重置 tap 计数
                        last_tap_coords = None
                        stuck_action_counts.clear()  # 新计划重置卡死计数

        except websockets.ConnectionClosed:
            print(f"[WS] 连接断开，任务 [{task_id}] 中止")
            self.store.update(task_id,
                              status=TaskStatus.FAILED,
                              error="WebSocket 连接断开",
                              completed_at=datetime.now().isoformat(),
                              usage=usage.to_dict())
        except Exception as e:
            print(f"[错误] {e}")
            self.store.update(task_id,
                              status=TaskStatus.FAILED,
                              error=str(e),
                              completed_at=datetime.now().isoformat(),
                              usage=usage.to_dict())
            raise
        finally:
            usage.print_final()
            usage_dict = usage.to_dict()

            current_record = self.store.get(task_id)
            if current_record and current_record.status == TaskStatus.RUNNING:
                if success:
                    # 任务正常完成（finish / report / 所有步骤执行完毕）
                    self.store.update(task_id,
                                      status=TaskStatus.COMPLETED,
                                      result=final_result or "任务完成",
                                      completed_at=datetime.now().isoformat(),
                                      usage=usage_dict)
                else:
                    # 走完 MAX_STEPS 仍未完成
                    print(f"\n[超限] 已达到最大步数 {max_steps} 步，任务未完成")
                    print(f"[超限] 完成步骤：{step_index}/{len(plan) if plan else '?'}")
                    self.store.update(task_id,
                                      status=TaskStatus.FAILED,
                                      error=f"已达最大步数 {max_steps} 步，完成 {step_index}/{len(plan) if plan else '?'} 步",
                                      completed_at=datetime.now().isoformat(),
                                      usage=usage_dict)

    async def _validate_report(self, task: str, content: str) -> tuple[bool, str, TokenUsage]:
        """
        校验 report 的内容是否真正回答了任务要求。
        返回 (是否有效, 缺少什么, token用量)
        """
        prompt = f"""你是一个结果验证专家。

用户的原始任务：
{task}

Agent 准备汇报的内容：
{content}

请完成以下两步判断：

第一步：理解任务的核心诉求
- 用户究竟想要什么具体信息？（是数据、列表、价格、地址、步骤，还是别的？）
- 期望的信息应该有什么特征？（有具体数字？有具体名称？有时间？）

第二步：对照评估汇报内容
- 汇报内容是否直接包含了用户需要的核心信息？
- 还是只有相关话题的边缘内容（如新闻报道、预测、摘要）而不是原始数据本身？

评估原则：
- 不要过度苛刻，核心信息存在即为合格，不必面面俱到
- 但如果用户要的是具体数值/列表，而内容只有模糊描述，则不合格
- 如果用户要的是操作结果（如"发送消息"），则有完成记录即合格
- 【重要】如果汇报内容明确说明了为什么某些数据无法获取（例如：比赛尚未开始/结束、搜索无结果、
  当前只有部分比赛已完成等），且对已有数据做了如实汇报，视为合格——不要因"不够完整"而拒绝
- 客观原因导致的数据缺失（赛事未开赛、数据暂未更新）不应成为拒绝的理由

只输出 JSON，不要其他文字：
{{"valid": true/false, "missing": "不合格时，用一句话说明缺少什么"}}"""

        validate_usage = TokenUsage()
        try:
            rsp, validate_usage = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            return data.get("valid", True), data.get("missing", ""), validate_usage
        except Exception as e:
            print(f"[校验] 解析失败（{e}），默认放行")
            return True, "", validate_usage

    async def _verify_operation(self, task: str, ui_text: str,
                                websocket=None, ui_element_count: int = 99) -> tuple[bool, str, TokenUsage]:
        """
        操作+验证类：通过当前界面文字判断操作是否真正成功完成。
        若文本验证失败且元素数量很少（WebView 场景，无障碍树看不到内容），
        自动降级为截图视觉验证。
        返回 (是否成功, 原因说明, token用量)
        """
        WEBVIEW_ELEM_THRESHOLD = 10  # 元素数低于此值视为 WebView，文本不可信

        prompt = f"""你是手机自动化验证专家。

用户的任务：{task}

当前界面元素：
{ui_text}

请判断：该操作是否已经成功完成并在界面上有明确体现？

判断标准：
- 必须在当前界面上看到操作成功的明确标志
  （如消息已出现在对话框、"发送成功"提示、点赞按钮状态改变、转账成功页面、预订确认单等）
- 仅仅"操作已执行"不代表成功，需要有可见的结果
- 如果界面只是回到了原始状态或没有任何成功标志，判为未完成

只输出 JSON：{{"success": true/false, "reason": "一句话说明判断依据"}}"""

        verify_op_usage = TokenUsage()
        try:
            rsp, verify_op_usage = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            text_success = data.get("success", False)
            text_reason = data.get("reason", "")

            # 文本验证通过，直接返回
            if text_success:
                return True, text_reason, verify_op_usage

            # 文本验证失败 + 元素极少（WebView 内容不在无障碍树里）→ 降级截图验证
            if websocket and ui_element_count < WEBVIEW_ELEM_THRESHOLD:
                print(f"[验证] 文本验证失败（{text_reason}），"
                      f"当前仅 {ui_element_count} 个元素（可能是 WebView），降级为截图验证...")
                try:
                    screenshot_b64 = await self._request_screenshot(websocket)
                    img_path = os.path.join(config.SCREENSHOT_DIR, "verify_vision.jpg")
                    _raw = base64.b64decode(screenshot_b64)
                    _PILImage.open(BytesIO(_raw)).convert("RGB").save(
                        img_path, format="JPEG", quality=85, optimize=True
                    )

                    vision_prompt = f"""你是手机自动化验证专家。

用户的任务：{task}

请根据截图判断：该操作是否已经成功完成并在界面上有明确体现？

判断标准：
- 必须在截图中看到操作成功的明确标志（成功提示、记录条目、状态变化等）
- 如果截图中可见相关成功记录（如打卡时间、发送的消息、点赞状态），即视为成功
- 如果截图只显示空白容器或无关内容，判为未完成

只输出 JSON：{{"success": true/false, "reason": "一句话说明判断依据"}}"""

                    vision_rsp, vision_usage = await self._in_thread(
                        self.vision_llm.predict, vision_prompt, img_path
                    )
                    verify_op_usage.prompt += vision_usage.prompt
                    verify_op_usage.completion += vision_usage.completion
                    verify_op_usage.cost += vision_usage.cost

                    vision_data = extract_json(vision_rsp)
                    vision_success = vision_data.get("success", False)
                    vision_reason = vision_data.get("reason", "")
                    print(f"[验证] 截图验证结果：{'✓' if vision_success else '✗'} {vision_reason}")
                    return vision_success, vision_reason, verify_op_usage
                except websockets.ConnectionClosed:
                    raise
                except Exception as e:
                    print(f"[验证] 截图验证失败（{e}），沿用文本验证结果")

            return False, text_reason, verify_op_usage
        except websockets.ConnectionClosed:
            raise
        except Exception as e:
            print(f"[验证] 解析失败（{e}），默认放行")
            return True, "", verify_op_usage

    def _build_partial_hint(self, action_log: list[dict], final_result: str | None) -> dict:
        """
        从部分成功的执行过程中提取经验，供下次任务规划参考。
        - failed_paths: 尝试过但无效的搜索/操作
        - found_info:   本次找到的部分内容
        - suggestion:   给 Planner 的建议
        """
        failed_paths = []
        for action in action_log:
            act = action.get("action", "")
            params = action.get("params", {})
            reason = action.get("reason", "")
            if act == "search_web":
                q = params.get("query", "")
                if q:
                    failed_paths.append(f"搜索：{q}")
            elif act == "click" and reason:
                # reason 里通常包含点击目标的描述
                failed_paths.append(f"点击：{reason[:40]}")

        # 去重 + 截取前 4 条
        seen = set()
        unique_paths = []
        for p in failed_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
            if len(unique_paths) >= 4:
                break

        # 根据失败路径推断通用建议
        if any("搜索" in p for p in unique_paths):
            suggestion = (
                "上次通过搜索引擎未找到完整结构化数据。"
                "建议尝试更精确的搜索关键词（加上具体日期、英文关键词或'统计'、'数据'等），"
                "或直接访问专业垂直网站（官网、权威数据平台等）。"
            )
        else:
            suggestion = (
                "上次执行路径遇到阻碍，建议尝试不同的操作路径或更换目标元素。"
            )

        return {
            "failed_paths": unique_paths,
            "found_info": (final_result or "")[:300] if final_result else "未找到有效信息",
            "suggestion": suggestion,
        }

    def _summarize_tried_approaches(self, history: list[str]) -> str:
        """
        从 history 中提炼已尝试过的关键操作路径（搜索词、点击目标等），
        格式化为文字供 Planner 在重规划时避开。
        """
        if not history:
            return ""
        # history 格式：["步骤N：xxx -> action -> 成功/失败", ...]
        lines = []
        for entry in history[-12:]:   # 最多回看12步
            lines.append(f"  - {entry}")
        return "\n".join(lines)

    def _save_report(self, task: str, content: str):
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.REPORTS_DIR, f"report_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"任务：{task}\n")
            f.write(f"时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(content)
        print("\n" + "=" * 50)
        print("📋 收集到的信息：")
        print("=" * 50)
        print(content)
        print("=" * 50)
        print(f"📁 已保存到：{path}")

    async def _request_state(self, websocket) -> dict:
        await websocket.send(json.dumps({"type": "request_state"}))
        raw = await websocket.recv()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[警告] _request_state 响应解析失败（{e}），返回空状态")
            return {"ui_elements": []}

    async def _request_screenshot(self, websocket) -> str:
        await websocket.send(json.dumps({"type": "request_screenshot"}))
        raw = await websocket.recv()
        try:
            return json.loads(raw).get("screenshot", "")
        except json.JSONDecodeError as e:
            print(f"[警告] _request_screenshot 响应解析失败（{e}），返回空字符串")
            return ""

    def _build_command(self, action: dict, ui_elements: list[dict],
                       screen_size: list[int] = None) -> dict | None:
        """
        构建发给手机的指令。
        若 index 越界或元素 bounds 无效（不可见），返回 None，由调用方标记失败并重试。
        screen_size: [width, height]，用于检测并修正靠近屏幕底部的点击坐标。
        """
        act = action.get("action")
        params = action.get("params", {})

        def get_valid_elem(target_idx: int, act_name: str) -> dict | None:
            """按 index 字段查找元素，并验证 bounds 有效。"""
            for e in ui_elements:
                if e.get("index") == target_idx:
                    if self.analyzer._valid_bounds(e):
                        return e
                    b = e.get("bounds", [])
                    print(f"[警告] {act_name} 元素 index={target_idx} bounds 无效 {b}，跳过（不可见元素）")
                    return None
            print(f"[警告] {act_name} 元素 index={target_idx} 不存在（共 {len(ui_elements)} 个）")
            return None

        def safe_y(y: int, label: str = "") -> int:
            """
            若 y 坐标落在屏幕底部 8% 安全区以内（可能与导航条/手势区重叠），
            自动上移至安全区上沿 - 20px。
            典型场景：TG "Share in 1 chat" 按钮 bounds 被无障碍树报告到导航区内。
            """
            if screen_size and screen_size[1] > 0:
                threshold = int(screen_size[1] * 0.92)
                if y > threshold:
                    adjusted = threshold - 20
                    print(f"[坐标安全]{' ' + label if label else ''} "
                          f"y={y} 超出安全区（>{threshold}），自动上移至 y={adjusted}")
                    return adjusted
            return y

        if act in ("click", "long_click"):
            elem = get_valid_elem(params.get("index", 1), act)
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            y = safe_y(y, f"click index={params.get('index', 1)}")
            return {"type": act, "x": x, "y": y}

        if act == "type":
            elem = get_valid_elem(params.get("index", 1), "type")
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            y = safe_y(y, "type")
            return {"type": "type", "x": x, "y": y, "text": params.get("text", "")}

        if act == "scroll":
            elem = get_valid_elem(params.get("index", 1), "scroll")
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            y = safe_y(y, "scroll")
            return {"type": "scroll", "x": x, "y": y,
                    "direction": params.get("direction", "up")}

        if act == "tap":
            # 直接坐标点击，用于 WebView/小程序中不在无障碍树里的按钮
            return {"type": "click", "x": int(params.get("x", 0)), "y": int(params.get("y", 0))}

        if act == "find_package":
            return {"type": "find_package", "keyword": params.get("keyword", "")}

        if act == "open_app":
            return {"type": "open_app", "package": params.get("package", "")}

        if act == "search_web":
            return {"type": "search_web", "query": params.get("query", "")}

        if act == "back":
            return {"type": "back"}

        if act == "home":
            return {"type": "home"}

        # 未知动作类型（LLM 幻觉）：拒绝执行，返回 None 触发重试，而非盲目转发给手机
        print(f"[警告] _build_command 遇到未知动作类型 '{act}'，跳过")
        return None
