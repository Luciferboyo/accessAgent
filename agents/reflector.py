import json
from models.llm import TextLLM, TokenUsage
from utils import extract_json

SYSTEM = """你是手机自动化验证专家。根据操作前后的界面变化，判断该步骤是否成功完成。

不同动作的判断标准：
- click / long_click / type：界面应出现预期变化（新页面、弹窗、内容更新等）
- scroll：界面内容发生滚动变化即为成功，无需到达特定位置
- back / home：界面切换到上一页或桌面即为成功
- open_app / search_web：出现对应应用或搜索页面即为成功

⚠️ 最重要的判断原则：评估【目标状态是否达成】，而非【是否使用了指定方法】
- 步骤描述中可能指定了某种方法（如"使用 search_web 搜索"、"点击地址栏"），
  但只要操作后界面已到达步骤目标所描述的状态，无论路径如何，都应返回 success=true
- 示例：步骤"使用 search_web 搜索骑士数据" → 实际通过点击标签页进入了骑士比赛数据页面
  → 已到达"获取骑士数据"的目标状态 → success=true（不要因为没用 search_web 就判失败）
- 示例：步骤"搜索关键词进入结果页" → 实际已直接进入了相关数据统计页面
  → 跳过了搜索步骤直接到达了更靠后的目标 → success=true

⚠️ 关键规则：步骤目标必须【完全达成】才能返回 success=true
- 如果步骤描述包含"所有"、"全部"、"逐一"、"每一个"等词，
  表示这一步需要多次操作才能完成，单次操作只是部分完成，应返回 success=false
  例：步骤"关闭所有非目标标签页"，只关了 1 个而还有其他非目标标签页未关 → success=false
- 如果步骤描述包含"确认"、"验证"字样，必须在界面上看到明确的成功标志才能返回 success=true
- success=true 表示本步骤目标已【完全】达成，可以进入下一步
- success=false 表示本步骤还需要继续操作（部分完成也算 false）

注意：
- 界面无变化 + 操作是 scroll，可能只是到底了，仍可视为成功
- 界面无变化 + 操作是 click，通常表示失败
- 不要因为没有获取到最终目标信息就判断失败，只要操作本身推进了进度就算成功

输出 JSON 格式（只输出 JSON，不要其他内容）：
{"success": true/false, "reason": "判断原因（若步骤目标未完全达成，说明还差什么）", "progress": "任务进展描述"}"""


class Reflector:
    def __init__(self, llm: TextLLM):
        self.llm = llm

    def verify(self, current_step: str, action_taken: dict,
               ui_before: str, ui_after: str) -> tuple[dict, TokenUsage]:

        action_type = action_taken.get("action", "")
        ui_changed = ui_before.strip() != ui_after.strip()

        # scroll 且界面无变化：可能已到底，仍算成功
        if action_type == "scroll" and not ui_changed:
            return (
                {"success": True, "reason": "scroll 后界面无变化，可能已到底/顶", "progress": "继续"},
                TokenUsage()
            )

        prompt = f"""当前步骤：{current_step}
执行的操作：{json.dumps(action_taken, ensure_ascii=False)}
界面是否发生变化：{"是" if ui_changed else "否"}

操作前界面：
{ui_before}

操作后界面：
{ui_after}

请判断该操作后本步骤目标是否达成。

判断要点：
1. 核心问题：操作后的界面状态，是否达到了步骤描述的目标？
2. 不要拘泥于步骤指定的"方法"（如 search_web、点击地址栏），只看"结果状态"
3. 如果当前界面已经到达了比当前步骤更靠后的目标状态（提前完成），也返回 success=true
4. 如果步骤要求"关闭所有/全部/逐一"等，判断操作后是否还存在未完成的目标"""

        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            return data, usage
        except Exception:
            return {"success": False, "reason": "验证解析失败", "progress": "未知"}, usage
