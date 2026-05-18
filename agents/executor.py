import json
from models.llm import TextLLM, VisionLLM, TokenUsage

TEXT_SYSTEM = """你是手机自动化执行专家。根据当前界面元素和当前步骤，决定下一个操作。

可用动作：
- click(index)：点击编号为 index 的元素
- long_click(index)：长按编号为 index 的元素
- type(index, text)：在编号为 index 的输入框输入文字
- scroll(index, direction)：在元素上滑动，direction 为 up/down/left/right
- back()：返回上一页
- home()：回到主屏幕
- open_app(package)：打开指定包名的应用
- search_web(query)：直接用 Chrome 搜索指定关键词（推荐用于网页搜索，比手动操作地址栏更可靠）
- finish()：任务已完成（仅用于纯操作类任务）
- report(content)：将收集到的信息完整汇报给用户（信息收集类任务必须用此结束，不能用finish）
- need_screenshot()：当前界面元素信息不足以判断，需要查看截图

⚠️ 重要规则：
- 需要在浏览器搜索内容时，优先使用 search_web 动作，不要手动操作地址栏
- 判断用 finish 还是 report：看任务的本质目标
  - 目标是"让用户知道某些信息"→ 必须用 report 汇报内容
  - 目标是"完成某个操作"（如发消息、打开应用、调整设置）→ 用 finish
- 只有从界面上读取到了完整的目标信息后，才能调用 report
- report 的 content 必须包含真实读取到的具体内容，不能是空话或假设性描述

输出 JSON 格式：
{"action": "click", "params": {"index": 3}, "reason": "点击搜索框"}
{"action": "type", "params": {"index": 2, "text": "搜索内容"}, "reason": "输入关键词"}
{"action": "report", "params": {"content": "查询结果：\\n..."}, "reason": "已收集到完整信息"}
{"action": "need_screenshot", "params": {}, "reason": "界面元素无文字，需要看截图确认"}"""

VISION_SYSTEM = """你是手机自动化执行专家。根据截图中的编号元素和当前步骤，决定下一个操作。

可用动作：
- click(index)：点击编号为 index 的元素
- long_click(index)：长按编号为 index 的元素
- type(index, text)：在编号为 index 的输入框输入文字
- scroll(index, direction)：在元素上滑动，direction 为 up/down/left/right
- back()：返回上一页
- home()：回到主屏幕
- open_app(package)：打开指定包名的应用
- search_web(query)：直接用 Chrome 搜索指定关键词（推荐用于网页搜索）
- finish()：任务已完成（仅用于纯操作类任务）
- report(content)：将收集到的信息完整汇报给用户（信息收集类任务必须用此结束）

⚠️ 重要规则：
- 需要搜索网页时，优先使用 search_web，不要手动操作地址栏
- 如果任务包含"搜索"、"查询"、"获取"、"告诉我"、"汇报"等字眼，必须先收集完整信息，再用 report 结束
- 不要在还没有获取目标信息时就调用 finish 或 report

⚠️ report 的严格标准：
- 内容必须直接回答用户的核心问题，不能用相关话题的边缘内容充数
- 先思考：用户真正想要的是什么？（具体数字？列表？名称？操作结果？）
- 再判断：截图里的内容是否完整包含了用户想要的那类信息？
- 如果截图只显示搜索结果列表、文章摘要、新闻标题，通常不够，需要点进详情页
- 宁可多操作几步获取完整数据，不可用不完整的内容提前汇报

输出 JSON 格式：
{"action": "click", "params": {"index": 3}, "reason": "点击搜索框"}
{"action": "report", "params": {"content": "以下是NBA季后赛赛程：\\n..."}, "reason": "已收集到完整信息"}"""


class Executor:
    def __init__(self, text_llm: TextLLM, vision_llm: VisionLLM):
        self.text_llm = text_llm
        self.vision_llm = vision_llm

    def decide_text(self, current_step: str, ui_text: str,
                    history: list[str], failure_reason: str = "") -> tuple[dict, TokenUsage]:
        history_text = "\n".join(history[-5:]) if history else "无"
        failure_hint = f"\n上一步失败原因：{failure_reason}" if failure_reason else ""

        prompt = f"""当前步骤：{current_step}

界面元素：
{ui_text}

最近操作历史：
{history_text}{failure_hint}

请决定下一个操作。如果界面元素信息不足以判断，请返回 need_screenshot。
如果已经收集到目标信息，请使用 report 动作汇报内容。"""

        rsp, usage = self.text_llm.predict(prompt, system=TEXT_SYSTEM)
        return self._parse(rsp), usage

    def decide_vision(self, current_step: str, annotated_image: str,
                      ui_text: str, history: list[str],
                      failure_reason: str = "") -> tuple[dict, TokenUsage]:
        history_text = "\n".join(history[-5:]) if history else "无"
        failure_hint = f"\n上一步失败原因：{failure_reason}" if failure_reason else ""

        prompt = f"""当前步骤：{current_step}

界面元素（辅助参考）：
{ui_text}

最近操作历史：
{history_text}{failure_hint}

请根据截图决定下一个操作。
如果已经收集到目标信息，请使用 report 动作汇报内容。"""

        rsp, usage = self.vision_llm.predict(prompt, annotated_image, system=VISION_SYSTEM)
        return self._parse(rsp), usage

    def _parse(self, rsp: str) -> dict:
        try:
            data = json.loads(rsp[rsp.find("{"):rsp.rfind("}") + 1])
            return data
        except Exception:
            return {"action": "need_screenshot", "params": {},
                    "reason": "文本决策解析失败，需要截图确认"}
