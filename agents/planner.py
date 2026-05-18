import json
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
- 不要合并多个操作到一个步骤中

输出 JSON 格式：{"steps": ["步骤1", "步骤2", ...]}

示例（发送消息任务）：
{"steps": [
  "打开微信，进入目标联系人对话框",
  "点击输入框，输入消息内容",
  "点击发送按钮",
  "确认消息已出现在对话框中，操作成功"
]}

示例（浏览器搜索任务）：
{"steps": [
  "使用 search_web 直接搜索目标关键词",
  "滚动查看搜索结果，找到目标信息",
  "将收集到的信息整理成列表汇报给用户"
]}"""


class Planner:
    def __init__(self, llm: TextLLM):
        self.llm = llm

    def make_plan(self, task: str, ui_text: str,
                  hint: dict = None) -> tuple[list[str], TokenUsage]:
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

        prompt = f"""任务：{task}
{hint_text}
当前界面元素：
{ui_text}

请拆解任务步骤。"""
        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            return data.get("steps", [task]), usage
        except Exception:
            return [task], usage

    def revise_plan(self, task: str, original_plan: list[str],
                    completed: int, failure_reason: str,
                    ui_text: str) -> tuple[list[str], TokenUsage]:
        prompt = f"""任务：{task}
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
