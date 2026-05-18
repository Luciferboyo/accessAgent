import asyncio
import functools
import json
import os
import re
import websockets
from dataclasses import dataclass
from datetime import datetime

from config import config
from utils import extract_json
from models.llm import TextLLM, VisionLLM, TokenUsage
from agents.planner import Planner
from agents.executor import Executor
from agents.reflector import Reflector
from memory.task_memory import TaskMemory
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
        self.planner = Planner(text_llm)
        self.executor = Executor(text_llm, vision_llm)
        self.reflector = Reflector(text_llm)
        self.memory = TaskMemory()
        self.analyzer = UIAnalyzer()
        self.annotator = ScreenAnnotator()

        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    @staticmethod
    async def _in_thread(fn, *args, timeout: int = 90):
        """在线程池中执行同步阻塞函数，不阻塞事件循环，超时自动抛出"""
        loop = asyncio.get_running_loop()
        coro = loop.run_in_executor(None, functools.partial(fn, *args))
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"LLM 调用超时（>{timeout}s），请检查网络或 API 状态")

    async def start(self, host: str, port: int):
        print(f"[WS] AccessAgent WebSocket 启动，监听 ws://{host}:{port}")
        async with websockets.serve(self.handle, host, port,
                                    max_size=10 * 1024 * 1024,
                                    ping_interval=None):
            await asyncio.Future()

    async def _classify_task(self, task: str) -> str:
        """
        用 LLM 判断任务类型，返回以下三种之一：
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

        try:
            rsp, _ = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            task_type = data.get("task_type", "operation")
            if task_type not in ("info", "operation", "verify"):
                task_type = "operation"
            label = {"info": "信息收集类", "operation": "纯操作类", "verify": "操作+验证类"}
            print(f"[任务分类] {label[task_type]} | {data.get('reason', '')}")
            return task_type
        except Exception as e:
            print(f"[任务分类] 解析失败（{e}），默认为 operation")
            return "operation"

    async def handle(self, websocket):
        print("[WS] Android App 已连接，等待任务...")

        # 从队列取下一个待执行任务
        task_id = await self.store.queue.get()
        record = self.store.get(task_id)
        if record is None:
            print(f"[错误] 任务 {task_id} 在 store 中不存在，跳过")
            return
        task = record.task

        print(f"[WS] 开始执行任务 [{task_id}]：{task}")
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

        # 任务开始时分类一次，后续复用，不重复调用 LLM
        # task_type: "info" | "operation" | "verify"
        task_type = await self._classify_task(task)

        current_state = await self._request_state(websocket)

        try:
            for global_step in range(config.MAX_STEPS):
                print(f"\n{'='*50}")
                print(f"[Step {global_step + 1}] [{task_id}]")

                # ── 1. 使用当前状态 ──────────────────────────────
                ui_elements = current_state.get("ui_elements", [])
                ui_text = self.analyzer.parse_elements(ui_elements)
                print(f"[AccessInfo] {len(ui_elements)} 个可交互元素")

                # ── 2. 首轮生成计划 ──────────────────────────────
                if plan is None:
                    print("[Planner] 生成任务计划...")
                    plan, plan_usage = await self._in_thread(
                        self.planner.make_plan, task, ui_text, planner_hint
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
                        print("[注意] 信息收集任务步骤已完成，但尚未汇报结果，追加汇报步骤")
                        if plan is None:
                            plan = []
                        plan.append("将目前收集到的所有信息整理后用 report 汇报给用户，如果信息不完整请说明原因")
                        failure_reason = "所有规划步骤已完成，现在必须用 report 汇报收集到的内容"
                        continue
                    print("[完成] 所有步骤执行完毕")
                    await websocket.send(json.dumps({"type": "finish", "message": "任务完成"}))
                    self.memory.save_flow(task, plan, action_log, quality="full")
                    success = True
                    break

                current_step = plan[step_index]
                print(f"[步骤 {step_index + 1}/{len(plan)}] {current_step}")

                # ── 3. 优先文本决策 ──────────────────────────────
                print("[Text] 尝试文本决策...")
                action, text_usage = await self._in_thread(
                    self.executor.decide_text, current_step, step_index, len(plan), ui_text, history, failure_reason
                )
                usage.add_text(text_usage)
                print(f"[Text] {action.get('action')} | {action.get('reason')}")
                print(f"[Token] 文本决策：{text_usage}")

                # ── 4. 判断是否需要截图 ──────────────────────────
                # 文本模型已决定 report/finish 时，不触发 Vision（避免被覆盖）
                text_action = action.get("action")
                if text_action in ("report", "finish"):
                    need_vision = False
                else:
                    need_vision = (
                        text_action == "need_screenshot"
                        or self.analyzer.needs_screenshot(ui_elements)
                        or consecutive_failures >= 2
                    )

                vision_usage = None
                if need_vision:
                    if action.get("action") == "need_screenshot":
                        trigger = "文本信息不足"
                    elif consecutive_failures >= 1:
                        trigger = f"上步失败（{failure_reason}）"
                    else:
                        trigger = "界面元素无有效文字"

                    print(f"[Vision] 触发截图，原因：{trigger}")
                    screenshot_b64 = await self._request_screenshot(websocket)
                    # annotator 做文件 IO + PIL 处理，放线程池避免阻塞事件循环
                    img_path = await self._in_thread(
                        self.annotator.save_screenshot,
                        screenshot_b64, config.SCREENSHOT_DIR, global_step
                    )
                    annotated = await self._in_thread(
                        self.annotator.annotate,
                        img_path, ui_elements,
                        img_path.replace(".png", "_labeled.png")
                    )
                    print(f"[Vision] 标注截图：{annotated}")
                    action, vision_usage = await self._in_thread(
                        self.executor.decide_vision,
                        current_step, step_index, len(plan), annotated, ui_text, history, failure_reason
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
                            verified, verify_reason = await self._verify_operation(task, ui_text)

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
                    # 纯操作任务不应以 report 结束，引导改为 finish
                    if task_type == "operation":
                        print("[拦截] 纯操作任务不应使用 report，引导改用 finish")
                        failure_reason = (
                            "这是一个纯操作任务，目标是完成操作而非收集信息。"
                            "请不要使用 report，确认操作已完成后直接调用 finish。"
                        )
                        consecutive_failures += 1
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
                        valid, missing = await self._validate_report(task, content)

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
                cmd = self._build_command(action, ui_elements)
                if cmd is None:
                    # index 越界，不发送指令，直接标记失败重试
                    print("[警告] 元素 index 越界，跳过本次指令，重新获取界面状态")
                    consecutive_failures += 1
                    total_failures += 1
                    failure_reason = f"AI 指定的元素编号不存在（共 {len(ui_elements)} 个元素），请重新选择"
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
                print(f"[App] 执行状态：{result.get('status')}")

                # ── 7. 获取新状态（复用为下一步）────────────────
                ui_text_before = ui_text
                current_state = await self._request_state(websocket)
                ui_text_after = self.analyzer.parse_elements(
                    current_state.get("ui_elements", [])
                )

                # ── 8. 自我反思（系统级动作直接跳过，节省 token）────
                SKIP_REFLECT = {"back", "home", "open_app", "search_web"}
                act_name = action.get("action")
                if act_name in SKIP_REFLECT:
                    verify = {"success": True, "reason": f"{act_name} 系统动作默认成功"}
                    reflect_usage = TokenUsage()
                    print(f"[Reflector] 跳过（{act_name} 系统动作）")
                else:
                    print("[Reflector] 自我反思...")
                    verify, reflect_usage = await self._in_thread(
                        self.reflector.verify, current_step, action, ui_text_before, ui_text_after
                    )
                    usage.add_text(reflect_usage)
                    print(f"[Reflector] 成功：{verify.get('success')} | {verify.get('reason')}")
                    print(f"[Token] 反思用量：{reflect_usage}")

                # ── 9. 本步 token 小计 ───────────────────────────
                step_total = text_usage.total + reflect_usage.total
                if need_vision and vision_usage:
                    step_total += vision_usage.total
                print(f"[Token] 本步合计：{step_total} | 累计：{usage.grand_total}")

                history.append(
                    f"步骤{step_index + 1}：{current_step} -> "
                    f"{action.get('action')} -> "
                    f"{'成功' if verify.get('success') else '失败'}"
                )

                if verify.get("success"):
                    step_index += 1
                    consecutive_failures = 0
                    failure_reason = ""
                else:
                    consecutive_failures += 1
                    total_failures += 1
                    failure_reason = verify.get("reason", "")
                    print(f"[失败] 累计失败 {total_failures} 次")

                    if total_failures >= config.MAX_TOTAL_FAILURES:
                        error_msg = f"累计失败{total_failures}次，{failure_reason}"
                        await websocket.send(json.dumps({
                            "type": "finish",
                            "message": f"任务失败：{error_msg}"
                        }))
                        self.store.update(task_id,
                                          status=TaskStatus.FAILED,
                                          error=error_msg,
                                          completed_at=datetime.now().isoformat())
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
                                              completed_at=datetime.now().isoformat())
                            break
                        print(f"[Planner] 连续失败，第 {replan_count} 次重新规划...")
                        plan, replan_usage = await self._in_thread(
                            self.planner.revise_plan,
                            task, plan, step_index, failure_reason, ui_text_after
                        )
                        usage.add_text(replan_usage)
                        print(f"[Token] 重规划用量：{replan_usage}")
                        step_index = 0        # 新计划从第 0 步开始
                        consecutive_failures = 0
                        failure_reason = ""

        except websockets.ConnectionClosed:
            print(f"[WS] 连接断开，任务 [{task_id}] 中止")
            self.store.update(task_id,
                              status=TaskStatus.FAILED,
                              error="WebSocket 连接断开",
                              completed_at=datetime.now().isoformat())
        except Exception as e:
            print(f"[错误] {e}")
            self.store.update(task_id,
                              status=TaskStatus.FAILED,
                              error=str(e),
                              completed_at=datetime.now().isoformat())
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
                    print(f"\n[超限] 已达到最大步数 {config.MAX_STEPS} 步，任务未完成")
                    print(f"[超限] 完成步骤：{step_index}/{len(plan) if plan else '?'}")
                    self.store.update(task_id,
                                      status=TaskStatus.FAILED,
                                      error=f"已达最大步数 {config.MAX_STEPS} 步，完成 {step_index}/{len(plan) if plan else '?'} 步",
                                      completed_at=datetime.now().isoformat(),
                                      usage=usage_dict)

    async def _validate_report(self, task: str, content: str) -> tuple[bool, str]:
        """
        校验 report 的内容是否真正回答了任务要求。
        返回 (是否有效, 缺少什么)
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

只输出 JSON，不要其他文字：
{{"valid": true/false, "missing": "不合格时，用一句话说明缺少什么"}}"""

        try:
            rsp, _ = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            return data.get("valid", True), data.get("missing", "")
        except Exception as e:
            print(f"[校验] 解析失败（{e}），默认放行")
            return True, ""

    async def _verify_operation(self, task: str, ui_text: str) -> tuple[bool, str]:
        """
        操作+验证类：通过当前界面文字判断操作是否真正成功完成。
        返回 (是否成功, 原因说明)
        """
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

        try:
            rsp, _ = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            return data.get("success", False), data.get("reason", "")
        except Exception as e:
            print(f"[验证] 解析失败（{e}），默认放行")
            return True, ""

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

        return {
            "failed_paths": unique_paths,
            "found_info": (final_result or "")[:300] if final_result else "未找到有效信息",
            "suggestion": (
                "上次通过搜索引擎获取的主要是新闻文章，未找到结构化数据。"
                "建议直接访问专业数据网站（如 NBA 官网、ESPN、basketball-reference 等），"
                "或使用更精确的搜索关键词（加上 'box score'、'stats' 等英文词）。"
            ),
        }

    def _save_report(self, task: str, content: str):
        import datetime as dt
        os.makedirs("./reports", exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"./reports/report_{timestamp}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"任务：{task}\n")
            f.write(f"时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    def _build_command(self, action: dict, ui_elements: list[dict]) -> dict | None:
        """
        构建发给手机的指令。
        若 index 越界或元素 bounds 无效（不可见），返回 None，由调用方标记失败并重试。
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

        if act in ("click", "long_click"):
            elem = get_valid_elem(params.get("index", 1), act)
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            return {"type": act, "x": x, "y": y}

        if act == "type":
            elem = get_valid_elem(params.get("index", 1), "type")
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            return {"type": "type", "x": x, "y": y, "text": params.get("text", "")}

        if act == "scroll":
            elem = get_valid_elem(params.get("index", 1), "scroll")
            if elem is None:
                return None
            x, y = self.analyzer.get_center(elem)
            return {"type": "scroll", "x": x, "y": y,
                    "direction": params.get("direction", "up")}

        if act == "open_app":
            return {"type": "open_app", "package": params.get("package", "")}

        if act == "search_web":
            return {"type": "search_web", "query": params.get("query", "")}

        return {"type": act}
