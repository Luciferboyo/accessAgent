import json
from datetime import date
from models.llm import TextLLM, TokenUsage
from utils import extract_json

SYSTEM = """你是一个手机自动化规划专家。
用户会给你一个任务描述和当前界面状态。
请将任务拆解为 3-5 个【里程碑式目标步骤】，每步描述"要达到什么状态"，而不是"具体点哪个按钮"。

什么是里程碑式步骤：
✅ 好："进入 Lark 考勤打卡页面"（执行器自己决定如何导航）
✅ 好："在结果页找到 Spurs 球员得分数据"（执行器自己决定点击/滚动/截图）
❌ 差："点击底部导航栏→点击 More→找到 Attendance→点击进入"（过度细化，界面变化就失效）

可用的特殊动作（步骤中可以提示执行器使用）：
- search_web(query)：直接用 Chrome 搜索，无需手动操作浏览器任何控件，是搜索网页的唯一正确方式
- open_app(XXX)：直接打开应用，执行器自动处理包名

规划规则（严格遵守）：
- 步骤数量控制在 3-5 步，禁止过度拆分单个目标为多个微操作步骤
- 【禁止】规划任何浏览器手动操作：点击地址栏、点击搜索框、在输入框输入文字、点击搜索按钮/回车——一律改用 search_web
- 【禁止】在计划中出现 find_package——包名查询由执行器自动完成，无需规划
- 【禁止】规划"滑动打开应用抽屉"、"搜索应用名"等手动打开应用的步骤——一律用 open_app
- 涉及信息收集时，最后一步必须是"将收集到的信息整理并汇报给用户"
- 涉及需要确认结果的操作（发消息、点赞、转账等），最后一步是"确认操作结果已在界面上体现"
- 纯操作类任务（打开应用、调整设置等）最后一步不要写"汇报给用户"

⚠️ 时效性数据搜索（体育赛事、股价、天气等）：
- 搜索词必须使用 prompt 中提供的【今天日期】，不要使用"今天"、"今日"等模糊词
- 优先英文关键词：体育统计用 "box score YYYY-MM-DD"，股票用 代码+"stock price"
- 体育球员数据：搜索 "[队名] box score [YYYY-MM-DD]" 可直接显示所有球员统计表格

输出 JSON 格式：{"steps": ["步骤1", "步骤2", ...]}

示例（打卡任务）：
{"steps": [
  "使用 open_app 打开 Lark",
  "进入 Lark 的考勤打卡（Attendance）页面",
  "点击 Clock Out 按钮完成打卡（WebView 中按钮可能需要 tap 坐标操作）",
  "确认打卡成功状态已在界面上体现"
]}

示例（体育数据查询，今天是2026-05-19）：
{"steps": [
  "使用 search_web 搜索 'Spurs box score 2026-05-19'",
  "在结果页找到马刺队球员 PTS 得分统计数据",
  "汇报马刺队最高得分球员及得分"
]}

示例（发消息任务）：
{"steps": [
  "使用 open_app 打开微信，进入目标联系人对话框",
  "输入消息内容并点击发送",
  "确认消息已出现在对话框中"
]}"""


class Planner:
    def __init__(self, llm: TextLLM):
        self.llm = llm

    @staticmethod
    def _today() -> str:
        return date.today().strftime("%Y-%m-%d")

    def make_plan(self, task: str, ui_text: str,
                  hint: dict = None,
                  screen_desc: str = "") -> tuple[list[str], TokenUsage]:
        # 有上次部分成功的经验时，拼入 prompt 让 Planner 避免重复走弯路
        hint_text = ""
        if hint:
            parts = []
            if hint.get("failed_paths"):
                paths_str = "、".join(hint["failed_paths"][:3])
                parts.append(f"- 上次已尝试但无效的路径：{paths_str}")
            if hint.get("found_info"):
                parts.append(f"- 上次找到的部分信息：{hint['found_info'][:200]}")
            if hint.get("suggestion"):
                parts.append(f"- 建议：{hint['suggestion']}")
            if parts:
                hint_text = "\n【上次经验参考（请避免重复相同弯路，尝试新路径）】\n" + "\n".join(parts) + "\n"

        # 定向截图生成的当前页面描述（帮助 Planner 了解起始位置，避免规划冗余导航步骤）
        screen_context = f"\n【当前页面状态】{screen_desc}\n" if screen_desc else ""

        today = self._today()
        prompt = f"""任务：{task}
今天日期：{today}（请在所有涉及时效性数据的搜索词中使用此日期，不要使用"今天"等模糊词）
{screen_context}{hint_text}
当前界面元素：
{ui_text}

请拆解任务步骤。如果【当前页面状态】显示已在任务相关页面，请直接从当前位置规划后续步骤，无需重复导航。"""
        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            return data.get("steps", [task]), usage
        except Exception:
            return [task], usage

    def revise_plan(self, task: str, original_plan: list[str],
                    completed: int, failure_reason: str,
                    ui_text: str,
                    tried_approaches: str = "",
                    screen_desc: str = "") -> tuple[list[str], TokenUsage]:
        today = self._today()

        tried_text = (
            "\n【已尝试但失败的操作记录（重新规划时必须避开这些路径，尝试完全不同的方式）】\n"
            + tried_approaches + "\n"
        ) if tried_approaches else ""

        screen_context = f"\n【当前页面状态】{screen_desc}\n" if screen_desc else ""

        prompt = f"""任务：{task}
今天日期：{today}
原计划：{json.dumps(original_plan, ensure_ascii=False)}
已完成步骤数：{completed}，当前卡住原因：{failure_reason}
{screen_context}{tried_text}
当前界面元素：
{ui_text}

请从【当前页面状态】出发，重新规划完成任务所需的剩余步骤。
- 如果原方法反复失败，必须换一种完全不同的路径（不同的搜索词、不同的网站、不同的导航方式）
- 不要重复【已尝试但失败的操作记录】中的方法
- 里程碑式步骤，3-5步即可，不要过细拆分"""
        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            return data.get("steps", original_plan[completed:]), usage
        except Exception:
            return original_plan[completed:], usage
