import json
import os
import tempfile
from datetime import datetime


MEMORY_FILE = "./memory/task_flows.json"


class TaskMemory:
    """
    记录成功完成的任务流程。
    分两种质量等级：
    - full：完全成功，下次直接复用计划
    - partial：部分成功（如强制放行），下次仅作为经验提示传给 Planner，不复用计划
    """

    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        self.flows: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Memory] 记忆文件损坏或读取失败（{e}），已重置为空")
                return {}
        return {}

    def _save(self):
        """原子写入：先写临时文件，再 os.replace，防止写入中断导致文件损坏"""
        dir_name = os.path.dirname(os.path.abspath(MEMORY_FILE))
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.flows, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, MEMORY_FILE)
        except OSError as e:
            print(f"[Memory] 记忆文件保存失败（{e}）")

    def _extract_keywords(self, task: str) -> list[str]:
        """
        提取关键词：
        - 中文：取长度 >= 2 的连续中文片段
        - 英文：按空格分词，长度 >= 2
        """
        keywords = []
        for word in task.split():
            if len(word) >= 2:
                keywords.append(word.lower())
        chinese_chars = ""
        for ch in task:
            if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
                chinese_chars += ch
            else:
                if len(chinese_chars) >= 2:
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
        """Jaccard 相似度（关键词交集/并集）"""
        kw1 = set(self._extract_keywords(task))
        kw2 = set(self._extract_keywords(stored_task))
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / len(kw1 | kw2)

    def find_similar(self, task: str, threshold: float = 0.85) -> dict | None:
        """
        查找相似任务的历史记录。

        返回值：
        - None：未找到相似任务
        - {"quality": "full",    "steps": [...]}        完全成功，直接复用计划
        - {"quality": "partial", "hint":  {...}}        部分成功，作为经验提示
        """
        best_score = 0.0
        best_data = None
        best_key = None

        for key, data in self.flows.items():
            score = self._similarity(task, data.get("task", key))
            if score > best_score:
                best_score = score
                best_data = data
                best_key = key

        if best_score < threshold or best_data is None:
            print(f"[Memory] 未找到相似任务（最高相似度 {best_score:.0%}），重新规划")
            return None

        quality = best_data.get("quality", "full")
        label = "完全成功" if quality == "full" else "部分成功"
        print(f"[Memory] 找到相似任务（相似度 {best_score:.0%}，{label}）：{best_key}")

        if quality == "full":
            steps = best_data.get("steps", [])
            return {"quality": "full", "steps": steps} if steps else None

        # partial：返回经验提示，不复用计划
        return {"quality": "partial", "hint": best_data.get("hint", {})}

    def save_flow(self, task: str, steps: list[str], actions: list[dict],
                  quality: str = "full", hint: dict = None):
        """
        保存任务流程。

        quality:
          "full"    完全成功，下次直接复用 steps
          "partial" 部分成功（强制放行），只保存 hint，不复用 steps
        hint:
          partial 时必须传入，包含 failed_paths / found_info / suggestion
        """
        key = task[:40]
        record = {
            "task": task,
            "quality": quality,
            "saved_at": datetime.now().isoformat(),
        }
        if quality == "full":
            record["steps"] = steps
            record["actions"] = actions
        else:
            record["hint"] = hint or {}

        self.flows[key] = record
        self._save()
        label = "完全成功" if quality == "full" else "部分成功（经验提示）"
        print(f"[Memory] 已保存任务流程（{label}）：{key}")
