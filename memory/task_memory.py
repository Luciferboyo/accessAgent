import json
import os
from datetime import datetime


MEMORY_FILE = "./memory/task_flows.json"


class TaskMemory:
    """
    记录成功完成的任务流程。
    下次遇到完全相同或高度相似的任务直接复用，减少 AI 调用。
    """

    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        self.flows: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.flows, f, ensure_ascii=False, indent=2)

    def _extract_keywords(self, task: str) -> list[str]:
        """
        提取关键词：
        - 中文：取长度 >= 2 的连续中文片段
        - 英文：按空格分词，长度 >= 2
        """
        keywords = []
        # 英文词
        for word in task.split():
            if len(word) >= 2:
                keywords.append(word.lower())
        # 中文2字词组（滑动窗口）
        chinese_chars = ""
        for ch in task:
            if '一' <= ch <= '鿿':
                chinese_chars += ch
            else:
                if len(chinese_chars) >= 2:
                    # 提取所有长度>=2的子串
                    for l in range(2, min(5, len(chinese_chars) + 1)):
                        for i in range(len(chinese_chars) - l + 1):
                            keywords.append(chinese_chars[i:i+l])
                chinese_chars = ""
        if len(chinese_chars) >= 2:
            for l in range(2, min(5, len(chinese_chars) + 1)):
                for i in range(len(chinese_chars) - l + 1):
                    keywords.append(chinese_chars[i:i+l])
        return list(set(keywords))

    def _similarity(self, task: str, stored_task: str) -> float:
        """
        计算两个任务的相似度（0~1）。
        用关键词交集占比衡量，需要超过阈值才认为是相似任务。
        """
        kw1 = set(self._extract_keywords(task))
        kw2 = set(self._extract_keywords(stored_task))
        if not kw1 or not kw2:
            return 0.0
        intersection = kw1 & kw2
        # Jaccard 相似度
        union = kw1 | kw2
        return len(intersection) / len(union)

    def find_similar(self, task: str, threshold: float = 0.85) -> list[str] | None:
        """
        查找相似任务的历史步骤。
        相似度需超过 threshold（默认 0.85）才复用。
        """
        best_score = 0.0
        best_steps = None
        best_key = None

        for key, data in self.flows.items():
            score = self._similarity(task, data.get("task", key))
            if score > best_score:
                best_score = score
                best_steps = data["steps"]
                best_key = key

        if best_score >= threshold:
            print(f"[Memory] 找到相似任务（相似度 {best_score:.0%}）：{best_key}")
            return best_steps

        print(f"[Memory] 未找到相似任务（最高相似度 {best_score:.0%}），重新规划")
        return None

    def save_flow(self, task: str, steps: list[str], actions: list[dict]):
        key = task[:40]
        self.flows[key] = {
            "task": task,
            "steps": steps,
            "actions": actions,
            "saved_at": datetime.now().isoformat(),
        }
        self._save()
        print(f"[Memory] 已保存任务流程：{key}")
