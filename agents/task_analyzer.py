from models.llm import TextLLM, TokenUsage
from utils import extract_json

SYSTEM = """你是一个手机自动化任务的前置分析专家。
在 Agent 执行任务前，你负责三件事：
1. 验证任务前提是否合理
2. 判断信息类任务能否直接从知识回答（无需实时搜索）
3. 为需要实时数据的任务提供搜索假设和优化建议

诚实原则：
- 对于实时数据（今天的比赛结果、实时股价、当前天气等），绝对不要捏造
- 只有真正确定的事实才能设为 can_answer_directly=true
- 有合理猜测但不确定时，放入 hypothesis，由手机搜索验证"""


class TaskAnalyzer:
    def __init__(self, llm: TextLLM):
        self.llm = llm

    def analyze(self, task: str, task_type: str, today: str) -> tuple[dict, TokenUsage]:
        """
        对任务进行前置分析。

        返回 dict：
          valid              : bool  - 任务前提是否合理可执行
          issue              : str   - 不合理的原因（valid=True 时为空）
          can_answer_directly: bool  - 信息类任务能否直接从 LLM 知识回答（无需实时数据）
          direct_answer      : str   - 直接回答的完整内容（can_answer_directly=False 时为空）
          hypothesis         : str   - 需要实时数据但有猜测时，给出最佳假设供手机验证参考
          search_hint        : str   - 给 Planner 的搜索优化建议（关键词、推荐网站等）
        """
        prompt = f"""任务：{task}
今天日期：{today}
任务类型：{task_type}（info=信息收集，operation=纯操作，verify=操作+验证）

请完成以下分析，输出严格为 JSON：

【前提验证】（所有任务类型）
判断任务前提是否合理：
- 是否有逻辑矛盾？（如"查明天比赛结果"——未来无法预测）
- 用户描述的事实是否有明显错误？
- 任务本身是否可以完成？
- 操作类任务通常默认合理（设备是否安装了 App 无法预判，直接标记合理）

【知识预判】（仅针对 info 类型任务，operation/verify 类型跳过此步）
- 能否从你的训练知识直接回答？（不依赖实时数据）
  → 合理标准：历史事实、通用知识、固定数据（如"最多冠军球队"、"水的沸点"）
  → 不合理：今天的比赛结果、当前股价、今天天气——这些必须实时查询
- 如果需要实时数据，是否有合理的背景知识可以作为假设供验证？
  → 例如：知道马刺今天打了雷霆，猜测 Wembanyama 是高分球员，可以作为验证假设
- 给出最优搜索建议（最佳关键词、推荐网站等）

输出 JSON（严格格式，不要添加任何注释或额外文字）：
{{
  "valid": true,
  "issue": "",
  "can_answer_directly": false,
  "direct_answer": "",
  "hypothesis": "",
  "search_hint": ""
}}"""

        # Bug fix: 预先初始化 usage，确保 LLM 调用成功但 JSON 解析失败时
        # 仍能返回真实的 token 用量，而不是归零的 TokenUsage()
        usage = TokenUsage()
        try:
            rsp, usage = self.llm.predict(prompt, system=SYSTEM)
            data = extract_json(rsp)
            # 确保所有字段存在
            result = {
                "valid": data.get("valid", True),
                "issue": data.get("issue", ""),
                "can_answer_directly": data.get("can_answer_directly", False),
                "direct_answer": data.get("direct_answer", ""),
                "hypothesis": data.get("hypothesis", ""),
                "search_hint": data.get("search_hint", ""),
            }
            # operation/verify 类型不应直接回答（必须手机操作）
            if task_type != "info":
                result["can_answer_directly"] = False
                result["direct_answer"] = ""
            return result, usage
        except Exception as e:
            print(f"[TaskAnalyzer] 解析失败（{e}），默认放行")
            return {
                "valid": True, "issue": "",
                "can_answer_directly": False, "direct_answer": "",
                "hypothesis": "", "search_hint": "",
            }, usage  # 返回真实用量（若 LLM 调用成功但 JSON 解析失败）
