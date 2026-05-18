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

    async def _classify_task(self, task: str) -> bool:
        """
        用 LLM 判断任务类型：
        - True  = 信息收集类（目标是让用户知道某些信息），必须用 report 结束
        - False = 纯操作类（目标是完成某个动作），用 finish 结束
        只调用一次，结果缓存在 handle() 里。
        """
        prompt = f"""判断下面这个手机自动化任务的类型：

任务：{task}

任务类型定义：
- 信息收集类：用户最终目的是"获得某些信息"（如查询数据、搜索内容、读取结果、告知用户某个值）
- 纯操作类：用户最终目的是"完成某个操作"（如发消息、打开应用、修改设置、拍照、转账）

注意：
- 有些任务可能先操作再收集信息，以最终目的为准
- 如果任务里有"告诉我"、"汇报"、"查询结果"等，通常是信息收集类
- 如果任务里有"发送"、"打开"、"设置"、"拍"等，通常是纯操作类

只输出 JSON：{{"is_info_task": true/false, "reason": "一句话说明判断依据"}}"""

        try:
            rsp, _ = await self._in_thread(self.text_llm.predict, prompt)
            data = extract_json(rsp)
            result = data.get("is_info_task", False)
            print(f"[任务分类] {'信息收集类' if result else '纯操作类'} | {data.get('reason', '')}")
            return result
        except Exception as e:
            print(f"[任务分类] 解析失败（{e}），默认为操作类")
            return False

    async def handle(self, websocket):
        print("[WS] Android App 已连接，等待任务...")

        # 从队列取下一个待执行任务
        task_id = await self.store.queue.get()
        record = self.store.get(task_id)
        task = record.task

        print(f"[WS] 开始执行任务 [{task_id}]：{task}")
        self.store.update(task_id, status=TaskStatus.RUNNING)

        await websocket.send(json.dumps({"type": "task", "task": task}))

        plan = self.memory.find_similar(task)
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

        # 任务开始时分类一次，后续复用，不重复调用 LLM
        is_info_task = await self._classify_task(task)

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
                        self.planner.make_plan, task, ui_text
                    )
                    usage.add_text(plan_usage)
                    # 生成失败时保底：把整个任务作为一个步骤
                    if not plan:
                        plan = [task]
                    print(f"[Planner] 计划步骤：{plan}")
                    print(f"[Token] 规划用量：{plan_usage}")

                if plan is None or step_index >= len(plan):
                    if is_info_task:
                        # 信息收集类任务走完所有计划步骤，但没有 report，强制补一步
                        print("[注意] 信息收集任务步骤已完成，但尚未汇报结果，追加汇报步骤")
                        plan.append("将目前收集到的所有信息整理后用 report 汇报给用户，如果信息不完整请说明原因")
                        failure_reason = "所有规划步骤已完成，现在必须用 report 汇报收集到的内容"
                        continue
                    print("[完成] 所有步骤执行完毕")
                    await websocket.send(json.dumps({"type": "finish", "message": "任务完成"}))
                    self.memory.save_flow(task, plan, action_log)
                    success = True
                    break

                current_step = plan[step_index]
                print(f"[步骤 {step_index + 1}/{len(plan)}] {current_step}")

                # ── 3. 优先文本决策 ──────────────────────────────
                print("[Text] 尝试文本决策...")
                action, text_usage = await self._in_thread(
                    self.executor.decide_text, current_step, ui_text, history, failure_reason
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
                        current_step, annotated, ui_text, history, failure_reason
                    )
                    usage.add_vision(vision_usage)
                    print(f"[Vision] {action.get('action')} | {action.get('reason')}")
                    print(f"[Token] 视觉决策：{vision_usage}")

                # ── 5. 任务完成或汇报结果 ────────────────────────
                if action.get("action") == "finish":
                    if is_info_task:
                        print("[拦截] 信息收集任务不允许 finish，继续执行...")
                        failure_reason = "任务需要收集信息并用 report 汇报结果，绝对不能用 finish 结束"
                        consecutive_failures += 1
                        current_state = await self._request_state(websocket)
                        continue
                    await websocket.send(json.dumps({"type": "finish", "message": "任务完成"}))
                    self.memory.save_flow(task, plan, action_log)
                    success = True
                    print("[完成] 任务成功结束")
                    break

                if action.get("action") == "report":
                    content = action.get("params", {}).get("content", "")

                    # report 被拒次数超过阈值：强制放行，接受最优努力结果
                    MAX_REPORT_REJECTIONS = 3
                    if report_rejected_count >= MAX_REPORT_REJECTIONS:
                        print(f"[放行] report 已被拒绝 {report_rejected_count} 次，强制接受当前内容")
                        valid, missing = True, ""
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
                    self.memory.save_flow(task, plan, action_log)
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
                result = json.loads(result_raw)
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
        return json.loads(raw).get("screenshot", "")

    def _build_command(self, action: dict, ui_elements: list[dict]) -> dict | None:
        """
        构建发给手机的指令。
        若 index 越界返回 None，由调用方标记失败并重试。
        """
        act = action.get("action")
        params = action.get("params", {})

        if act in ("click", "long_click"):
            idx = params.get("index", 1) - 1
            if 0 <= idx < len(ui_elements):
                x, y = self.analyzer.get_center(ui_elements[idx])
                return {"type": act, "x": x, "y": y}
            print(f"[警告] {act} 元素 index={idx+1} 越界（共 {len(ui_elements)} 个）")
            return None

        if act == "type":
            idx = params.get("index", 1) - 1
            if 0 <= idx < len(ui_elements):
                x, y = self.analyzer.get_center(ui_elements[idx])
                return {"type": "type", "x": x, "y": y, "text": params.get("text", "")}
            print(f"[警告] type 元素 index={idx+1} 越界（共 {len(ui_elements)} 个）")
            return None

        if act == "scroll":
            idx = params.get("index", 1) - 1
            if 0 <= idx < len(ui_elements):
                x, y = self.analyzer.get_center(ui_elements[idx])
                return {"type": "scroll", "x": x, "y": y,
                        "direction": params.get("direction", "up")}
            print(f"[警告] scroll 元素 index={idx+1} 越界（共 {len(ui_elements)} 个）")
            return None

        if act == "open_app":
            return {"type": "open_app", "package": params.get("package", "")}

        if act == "search_web":
            return {"type": "search_web", "query": params.get("query", "")}

        return {"type": act}
