import json
from datetime import date
from models.llm import TextLLM, TokenUsage
from utils import extract_json

SYSTEM = """你是一个手机自动化规划专家。
用户会给你一个任务描述和当前界面的 UI 元素信息。
请将任务拆解为有序的子步骤列表，每个步骤要具体明确，不能跳步。

可用的特殊动作（执行器支持）：
- search_web(query)：直接用 Chrome 打开搜索页面，无需手动操作地址栏，是浏览器搜索的首选方式

规划规则：
- 每个步骤只描述一个操作
- 【重要】涉及网页搜索时，必须使用 search_web 动作直接搜索，禁止规划"点击地址栏"、"输入网址"等手动步骤
- 涉及信息收集时，最后一步必须是"将收集到的信息整理并汇报给用户"
- 涉及需要确认结果的操作（发消息、点赞、转账、评论、预订等），最后一步必须是"确认操作结果已在界面上体现"
- 【重要】纯操作类任务（打开应用、删除页面、调整设置等）最后一步不要写"汇报给用户"，操作完成即结束，无需汇报
- 【重要】需要对多个目标逐一操作时（如"关闭所有非目标标签页"），必须拆分成多个单步骤，每步只操作一个目标
  例：不要写"逐一关闭所有X" → 应写"关闭第一个非目标标签页"、"关闭下一个非目标标签页"……或合并写"关闭一个非目标标签页（执行器将重复此步直到全部关闭）"
- 不要合并多个操作到一个步骤中
- 【重要】需要打开某个 App 时，直接规划"使用 open_app 打开 XXX"，执行器会自动处理包名查询
  - 禁止在计划中出现 find_package 步骤——包名查询由执行器自动完成，无需规划
  - 禁止规划"滑动打开应用抽屉"、"搜索应用名"等手动步骤

⚠️ 时效性数据搜索规则（体育赛事、股价、天气、新闻等）：
- 搜索词中必须包含 prompt 中提供的【今天日期】，不要使用模糊词"今天"、"今日"，否则搜索结果可能是历史数据
- 优先使用英文关键词组合搜索专业数据：
  - 体育比赛统计：用 "box score YYYY-MM-DD"、"game stats" 等，会直接显示结构化数据表格
  - 股票/指数：用 股票代码 + "stock price"
  - 天气：用 城市名 + "weather forecast"
- 避免只用中文通用词（如"NBA骑士比赛数据"），这类词容易进入直播赛程/综合门户站而非统计数据页
- 对于体育球员数据：搜索 "[队名] box score [YYYY-MM-DD]" 可直接跳到包含所有球员数据的统计页，无需逐个球员访问

输出 JSON 格式：{"steps": ["步骤1", "步骤2", ...]}

示例（发送消息任务）：
{"steps": [
  "打开微信，进入目标联系人对话框",
  "点击输入框，输入消息内容",
  "点击发送按钮",
  "确认消息已出现在对话框中，操作成功"
]}

示例（体育数据查询任务，今天是2026-05-18）：
{"steps": [
  "使用 search_web 搜索 'Cavaliers box score 2026-05-18'",
  "在搜索结果页面找到骑士队比赛的球员数据表格",
  "滚动查看完整球员统计数据（得分、篮板、助攻等）",
  "将所有球员的比赛数据整理后汇报给用户"
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
                    ui_text: str) -> tuple[list[str], TokenUsage]:
        today = self._today()
        prompt = f"""任务：{task}
今天日期：{today}
原计划：{json.dumps(original_plan, ensure_ascii=False)}
已完成步骤数：{completed}
失败原因：{failure_reason}

当前界面元素：
{ui_text}

请重新规划剩余步骤。"""
        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            return data.get("steps", original_plan[completed:]), usage
        except Exception:
            return original_plan[completed:], usage
