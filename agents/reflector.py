import json
from models.llm import TextLLM, TokenUsage
from utils import extract_json

SYSTEM = """你是手机自动化验证专家。当前任务使用里程碑式步骤规划（每个步骤描述一个目标状态，通常需要多次操作才能完全达成）。

你的职责：判断【刚才这一次操作】对当前里程碑步骤的推进效果，返回三种状态之一。

✅ done（里程碑完成）
- 当前步骤所描述的目标状态已在界面上完全体现
- 步骤中所有要求的条件均已满足
- 可以进入下一个步骤了
- ⚠️ 硬性要求：done 必须伴随操作后界面有可见变化（跳转新页面/新元素出现/元素消失/文字改变等）
  如果操作后界面完全无变化，则不能判为 done，即使"前置条件已满足"

🔄 progress（有效推进，尚未完成）
- 这次操作让我们明显向步骤目标方向前进了一步
- 但步骤的完整目标状态还未达成，还需要继续操作
- 典型情形：
  ✓ 目标"进入聊天并找到图片消息" → 操作后成功进入了聊天界面（还需滚动找图片）→ progress
  ✓ 目标"完成打卡" → 成功打开了考勤页面（还需点击打卡按钮）→ progress
  ✓ 目标"转发图片" → 长按图片后弹出了操作菜单（还需点击转发并选接收方）→ progress
  ✓ 目标"搜索并查看数据" → 搜索结果页已出现（还需点进去看详情）→ progress

❌ stuck（无效/卡住）
- 操作后 UI 几乎没有变化（非 scroll 操作）
- 操作后进入了与步骤目标无关的页面（走错方向）
- 操作后退回到了更早的状态（倒退）
- 在错误的应用/页面内做了操作（目标是 A 应用，却在 B 应用里操作）

⚠️ click/long_click 操作界面无变化的强制规则（最高优先级）：
- 如果操作是 click 或 long_click，且界面【没有任何变化】
  → 无论前置条件是否满足，必须判为 stuck
  → reason 说明"点击无响应，界面无变化"
  → next_hint 必须建议：改用 tap(x,y) 点击该按钮的截图视觉中心（该按钮可能在 WebView 或自定义渲染中，click(index) 无法触达）
- 这条规则凌驾于"前置条件已满足 → done"逻辑之上，不得例外

判断要点：
1. 只看"结果状态"是否在推进目标，不要拘泥于步骤中指定的"方法"
2. 如果当前界面已跳过当前步骤直接到达了更靠后的目标状态 → done（但仍需界面有变化）
3. 部分完成 → progress；完全没动或走错 → stuck

输出 JSON（只输出 JSON，不要其他内容）：
{"status": "done/progress/stuck", "reason": "一句话说明判断原因", "next_hint": "执行器下一步应做什么（done 时留空；progress/stuck 时必须给出具体可操作的建议，stuck 时尤其要说明应改用什么替代方法）"}"""


class Reflector:
    def __init__(self, llm: TextLLM):
        self.llm = llm

    def verify(self, current_step: str, action_taken: dict,
               ui_before: str, ui_after: str) -> tuple[dict, TokenUsage]:

        action_type = action_taken.get("action", "")
        ui_changed = ui_before.strip() != ui_after.strip()

        # scroll 特殊处理：不调用 LLM（节省 token）
        # 里程碑模式下，scroll 本身极少能"完成"一个步骤，统一返回 progress
        # - scroll + 无变化 → 已到列表边界，需截图确认内容
        # - scroll + 有变化 → 内容更新，继续推进
        if action_type == "scroll":
            if not ui_changed:
                return (
                    {
                        "status": "progress",
                        "reason": "scroll 后界面无变化，已到列表底部/顶部",
                        "next_hint": "已到达列表边界，可截图确认当前可见内容是否满足步骤目标"
                    },
                    TokenUsage()
                )
            else:
                return (
                    {
                        "status": "progress",
                        "reason": "scroll 后内容已更新，继续推进步骤",
                        "next_hint": ""
                    },
                    TokenUsage()
                )

        # click/long_click + 界面无变化 → 强制提示 LLM 判为 stuck
        no_change_warning = ""
        if action_type in ("click", "long_click") and not ui_changed:
            no_change_warning = (
                "\n⚠️ 注意：本次操作是 click/long_click 且界面完全无变化。"
                "根据强制规则，此情形必须判为 stuck（不得判为 done，即使前置条件已满足）。"
                "next_hint 请说明改用 tap(x,y) 点击该按钮的截图视觉中心。\n"
            )

        prompt = f"""当前里程碑步骤：{current_step}
执行的操作：{json.dumps(action_taken, ensure_ascii=False)}
界面是否发生变化：{"是" if ui_changed else "否"}{no_change_warning}
操作前界面：
{ui_before}

操作后界面：
{ui_after}

请判断该操作对当前里程碑步骤的推进效果（done/progress/stuck）。"""

        rsp, usage = self.llm.predict(prompt, system=SYSTEM)

        try:
            data = extract_json(rsp)
            # 兼容旧格式（success: true/false）
            if "status" not in data and "success" in data:
                data["status"] = "done" if data["success"] else "stuck"
            if "status" not in data:
                data["status"] = "stuck"
            if "next_hint" not in data:
                data["next_hint"] = ""
            return data, usage
        except Exception:
            return {"status": "stuck", "reason": "验证解析失败", "next_hint": ""}, usage
