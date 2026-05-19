import json
from models.llm import TextLLM, VisionLLM, TokenUsage
from utils import extract_json

TEXT_SYSTEM = """你是手机自动化执行专家。根据当前界面元素和当前步骤，决定下一个操作。

可用动作：
- click(index)：点击编号为 index 的元素
- long_click(index)：长按编号为 index 的元素
- type(index, text)：在编号为 index 的输入框输入文字
- scroll(index, direction)：在元素上滑动，direction 为 up/down/left/right
- back()：返回上一页
- home()：回到主屏幕
- find_package(keyword)：查询设备上安装的包名。用法：不知道包名时先调用此动作，系统会把结果反馈给你，然后你再用 open_app 打开（两次决策完成一个"打开应用"目标，find_package 后不要做任何其他操作）
- open_app(package)：打开指定包名的应用（必须使用设备上真实存在的包名，不能猜测）
- search_web(query)：直接用 Chrome 搜索指定关键词（推荐用于网页搜索，比手动操作地址栏更可靠）
- finish()：整个任务的所有目标都已完成
- report(content)：将收集到的信息完整汇报给用户（信息收集类任务必须用此结束，不能用finish）
- need_screenshot()：当前界面元素信息不足以判断，需要查看截图

⚠️ 关于 finish 的严格规定：
- finish 表示【整个原始任务】全部完成，而不是当前子步骤完成
- 提示中会告诉你"第 X 步 / 共 Y 步"，只有在最后几步且确认任务目标已全部实现时才能调用 finish
- 【禁止】仅因为当前子步骤的目标已满足就调用 finish——后续还有步骤时必须继续执行
- 如果当前步骤目标已由界面自动满足（无需操作），应执行一个推进性动作（如 scroll 确认、点击下一目标）推进到下一步

⚠️ 搜索优先级规则（非常重要）：
- 需要搜索网页时，【必须】使用 search_web 动作，严禁手动点击地址栏/输入/按回车
- 手动操作地址栏极易失败（点击→输入→提交 需要三步且每步都可能失败）
- search_web 是系统级原子操作，一步完成，成功率远高于手动操作
- 即使当前界面显示的是浏览器地址栏，也应调用 search_web 而非 type

⚠️ 识别"已提前到达目标"：
- 如果当前界面已经显示了任务后续步骤需要的内容（如数据统计页、目标网站），
  无需按计划步骤顺序操作，应直接从当前页面继续推进任务
- 例：计划第2步是"点击搜索结果"，但当前界面已经打开了相关数据页面
  → 跳过第2步，直接在当前页面执行第3步（滚动/读取数据）

⚠️ 其他重要规则：
- 判断用 finish 还是 report：看任务的本质目标
  - 目标是"让用户知道某些信息"→ 必须用 report 汇报内容
  - 目标是"完成某个需要确认结果的操作"（发消息、点赞、转账）→ 界面出现成功标志后才能 finish
  - 目标是"完成某个操作"（打开应用、调整设置、删除页面）→ 完成后用 finish
- 只有从界面上读取到了完整的目标信息后，才能调用 report
- report 的 content 必须包含真实读取到的具体内容，不能是空话或假设性描述

输出 JSON 格式：
{"action": "click", "params": {"index": 3}, "reason": "点击搜索框"}
{"action": "type", "params": {"index": 2, "text": "搜索内容"}, "reason": "输入关键词"}
{"action": "report", "params": {"content": "查询结果：\\n..."}, "reason": "已收集到完整信息"}
{"action": "need_screenshot", "params": {}, "reason": "界面元素无文字，需要看截图确认"}"""

VISION_SYSTEM = """你是手机自动化执行专家。根据截图中的编号元素和当前步骤，决定下一个操作。

可用动作：
- click(index)：点击编号为 index 的元素（元素在截图中有彩色方框和编号）
- long_click(index)：长按编号为 index 的元素
- type(index, text)：在编号为 index 的输入框输入文字
- scroll(index, direction)：在元素上滑动，direction 为 up/down/left/right
- tap(x, y)：直接点击手机屏幕的绝对坐标（手机像素），专用于 WebView/小程序中没有编号的按钮
  ⚠️ 换算方法：提示中会提供截图尺寸和手机分辨率，将截图中的像素坐标乘以换算比例即可得到手机坐标
- back()：返回上一页
- home()：回到主屏幕
- find_package(keyword)：查询设备上安装的包名（不知道包名时先查询，结果反馈后再用 open_app 打开）
- open_app(package)：打开指定包名的应用（必须使用设备上真实存在的包名）
- search_web(query)：直接用 Chrome 搜索指定关键词（推荐用于网页搜索）
- finish()：整个任务的所有目标都已完成
- report(content)：将收集到的信息完整汇报给用户（信息收集类任务必须用此结束）

⚠️ 关于 finish 的严格规定：
- finish 表示【整个原始任务】全部完成，而不是当前子步骤完成
- 提示中会告诉你"第 X 步 / 共 Y 步"，只有在最后几步且确认任务目标已全部实现时才能调用 finish
- 【禁止】仅因为当前子步骤的目标已满足就调用 finish——后续还有步骤时必须继续执行
- 如果当前步骤目标已由界面自动满足（无需操作），应执行一个推进性动作推进到下一步

⚠️ 搜索优先级规则（非常重要）：
- 需要搜索网页时，【必须】使用 search_web 动作，严禁手动点击地址栏/输入文字/按回车
- 手动操作地址栏极易失败，search_web 是系统级原子操作，一步完成
- 即使截图显示的是浏览器地址栏输入状态，也应直接调用 search_web

⚠️ 识别"已提前到达目标"：
- 如果截图已经显示了任务后续步骤需要的内容（如数据统计页、目标网站已打开），
  无需按计划顺序操作，直接在当前页面继续推进任务（滚动、读取数据等）
- 不要因为"计划说要先搜索"就放弃已经到达的正确页面重新搜索

⚠️ 其他重要规则：
- 如果任务包含"搜索"、"查询"、"获取"、"告诉我"、"汇报"等字眼，必须先收集完整信息，再用 report 结束
- 不要在还没有获取目标信息时就调用 finish 或 report
- 对于发消息、点赞、转账等操作，必须在截图中看到明确的成功标志才能 finish

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

    def decide_text(self, current_step: str, step_index: int, total_steps: int,
                    ui_text: str, history: list[str],
                    failure_reason: str = "",
                    consecutive_failures: int = 0) -> tuple[dict, TokenUsage]:
        history_text = "\n".join(history[-8:]) if history else "无"
        remaining = total_steps - step_index - 1
        progress = f"第 {step_index + 1} 步 / 共 {total_steps} 步，完成后还剩 {remaining} 步"

        if consecutive_failures >= 2:
            failure_hint = (
                f"\n\n🚨 当前方法已连续失败 {consecutive_failures} 次，必须换用完全不同的方法！"
                f"\n失败原因及建议：{failure_reason}"
            )
        elif failure_reason:
            failure_hint = f"\n\n上一步失败原因：{failure_reason}"
        else:
            failure_hint = ""

        prompt = f"""当前步骤（{progress}）：{current_step}

界面元素：
{ui_text}

最近操作历史：
{history_text}{failure_hint}

请决定下一个操作。如果界面元素信息不足以判断，请返回 need_screenshot。
如果已经收集到目标信息，请使用 report 动作汇报内容。
注意：还剩 {remaining} 步未执行，除非整个任务已提前完成，否则不要调用 finish。"""

        rsp, usage = self.text_llm.predict(prompt, system=TEXT_SYSTEM)
        return self._parse(rsp), usage

    def decide_vision(self, current_step: str, step_index: int, total_steps: int,
                      annotated_image: str, ui_text: str, history: list[str],
                      failure_reason: str = "",
                      screen_size: list[int] = None,
                      img_size: tuple[int, int] = None,
                      consecutive_failures: int = 0) -> tuple[dict, TokenUsage]:
        history_text = "\n".join(history[-8:]) if history else "无"
        remaining = total_steps - step_index - 1
        progress = f"第 {step_index + 1} 步 / 共 {total_steps} 步，完成后还剩 {remaining} 步"

        if consecutive_failures >= 2:
            failure_hint = (
                f"\n\n🚨 当前方法已连续失败 {consecutive_failures} 次，必须换用完全不同的方法！"
                f"\n失败原因及建议：{failure_reason}"
            )
        elif failure_reason:
            failure_hint = f"\n\n上一步失败原因：{failure_reason}"
        else:
            failure_hint = ""

        # 坐标换算提示：帮助 Vision 模型使用 tap(x,y) 点击 WebView 内的无编号按钮
        if screen_size and img_size and img_size[0] > 0:
            scale_x = screen_size[0] / img_size[0]
            scale_y = screen_size[1] / img_size[1]
            coord_hint = (
                f"\n【坐标参考】截图尺寸 {img_size[0]}×{img_size[1]}px，"
                f"手机分辨率 {screen_size[0]}×{screen_size[1]}px，"
                f"换算比例 ×{scale_x:.2f}（x）×{scale_y:.2f}（y）。"
                f"若目标按钮在截图中没有编号（WebView/小程序内容），"
                f"请用 tap(x,y)：手机x = 截图x×{scale_x:.2f}，手机y = 截图y×{scale_y:.2f}"
            )
        else:
            coord_hint = ""

        prompt = f"""当前步骤（{progress}）：{current_step}

界面元素（辅助参考）：
{ui_text}

最近操作历史：
{history_text}{failure_hint}{coord_hint}

请根据截图决定下一个操作。
- 如果目标按钮有编号，使用 click(index)
- 如果目标按钮没有编号（WebView/小程序中），使用 tap(x,y) 并按上方换算比例计算坐标
如果已经收集到目标信息，请使用 report 动作汇报内容。
注意：还剩 {remaining} 步未执行，除非整个任务已提前完成，否则不要调用 finish。"""

        rsp, usage = self.vision_llm.predict(prompt, annotated_image, system=VISION_SYSTEM)
        return self._parse(rsp), usage

    def _parse(self, rsp: str) -> dict:
        try:
            return extract_json(rsp)
        except Exception:
            return {"action": "need_screenshot", "params": {},
                    "reason": "文本决策解析失败，需要截图确认"}
