import hashlib
import json
import os
import tempfile
from datetime import datetime

from config import config

MEMORY_FILE = "./memory/task_flows.json"


class TaskMemory:
    """
    记录成功完成的任务流程。
    分两种质量等级：
    - full：完全成功，下次直接复用计划
    - partial：部分成功（如强制放行），下次仅作为经验提示传给 Planner，不复用计划

    每条记忆还记录以下元数据（自动维护）：
    - step_count：步骤数，用于保优策略（同质量保留步骤更少的路径）
    - use_count：被命中并返回的次数
    - last_used_at：最近一次被命中的时间
    - saved_at：首次保存时间
    - updated_at：最近一次更新时间（质量升级或步骤优化时刷新）
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

    def find_similar(self,
                     task: str,
                     full_threshold: float | None = None,
                     partial_threshold: float | None = None) -> dict | None:
        """
        查找相似任务的历史记录，并自动更新命中记忆的使用统计。

        阈值说明（均从 config 读取默认值）：
        - full_threshold    相似度 >= 此值且原记录为 full → 直接复用计划步骤（极高置信度）
        - partial_threshold 相似度 >= 此值但 < full_threshold → 仅提供经验提示，不复用计划
                            原为 full 质量的记录会被降级为 partial hint（包含步骤摘要建议）

        返回值：
        - None：未找到足够相似的任务
        - {"quality": "full",    "steps": [...]}   直接复用计划
        - {"quality": "partial", "hint":  {...}}   仅作为经验提示
        """
        if full_threshold is None:
            full_threshold = config.MEMORY_FULL_THRESHOLD
        if partial_threshold is None:
            partial_threshold = config.MEMORY_PARTIAL_THRESHOLD

        best_score = 0.0
        best_data = None
        best_key = None

        for key, data in self.flows.items():
            score = self._similarity(task, data.get("task", key))
            if score > best_score:
                best_score = score
                best_data = data
                best_key = key

        if best_score < partial_threshold or best_data is None:
            print(f"[Memory] 未找到相似任务（最高相似度 {best_score:.0%}），重新规划")
            return None

        quality = best_data.get("quality", "full")

        # ── 高置信度：直接复用计划 ──────────────────────────────────
        if best_score >= full_threshold and quality == "full":
            steps = best_data.get("steps", [])
            if not steps:
                return None
            step_count = len(steps)
            use_count = best_data.get("use_count", 0) + 1
            print(f"[Memory] 找到相似任务（相似度 {best_score:.0%}，完全成功，"
                  f"{step_count} 步，已复用 {use_count} 次）：{best_key}")
            # 更新使用统计
            self.flows[best_key]["use_count"] = use_count
            self.flows[best_key]["last_used_at"] = datetime.now().isoformat()
            self._save()
            return {"quality": "full", "steps": steps}

        # ── 中等置信度：降级为经验提示，不复用计划 ─────────────────
        # 原因：相似度未达 full_threshold，参数（用户名/金额/内容等）可能不同，
        # 直接复用计划风险较高，仅提取经验作为 Planner 参考。
        orig_label = "完全成功" if quality == "full" else "部分成功"
        use_count = best_data.get("use_count", 0) + 1
        print(f"[Memory] 找到相似任务（相似度 {best_score:.0%}，{orig_label}，"
              f"相似度未达 {full_threshold:.0%}，降级为经验提示，已复用 {use_count} 次）：{best_key}")

        # 更新使用统计
        self.flows[best_key]["use_count"] = use_count
        self.flows[best_key]["last_used_at"] = datetime.now().isoformat()
        self._save()

        hint = best_data.get("hint", {})
        if not hint and quality == "full":
            # 原 full 记录没有 hint 字段，从步骤列表提取摘要作为建议
            steps = best_data.get("steps", [])
            hint = {
                "suggestion": (
                    f"类似历史任务的参考步骤（仅供参考，请根据当前任务调整）："
                    f"{' → '.join(steps[:4])}"
                ),
                "failed_paths": [],
                "found_info": "",
            }
        return {"quality": "partial", "hint": hint}

    def save_flow(self, task: str, steps: list[str], actions: list[dict],
                  quality: str = "full", hint: dict = None):
        """
        保存任务流程，附带保优策略和质量升级逻辑。

        quality:
          "full"    完全成功，下次直接复用 steps
          "partial" 部分成功（强制放行），只保存 hint，不复用 steps
        hint:
          partial 时必须传入，包含 failed_paths / found_info / suggestion

        保优策略（防止用差路径覆盖好路径）：
        - full → full：仅当新步骤数 <= 旧步骤数时才覆盖（保留最短路径）
        - partial → full：无条件升级（质量提升优先）
        - full → partial：拒绝降级（已有更好的记录）
        - partial → partial：直接覆盖（hint 信息可能更新）
        """
        key = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        now = datetime.now().isoformat()
        existing = self.flows.get(key)
        new_step_count = len(steps) if quality == "full" else 0

        # ── 检查是否需要更新 ────────────────────────────────────────
        if existing is not None:
            old_quality = existing.get("quality", "full")
            old_step_count = existing.get("step_count", len(existing.get("steps", [])))

            # full → partial：拒绝降级
            if old_quality == "full" and quality == "partial":
                print(f"[Memory] 拒绝降级（已有 full 记录，步骤数 {old_step_count}），跳过保存：{key}")
                return

            # full → full：仅当新路径更短或相等时才覆盖
            if old_quality == "full" and quality == "full":
                if new_step_count > old_step_count:
                    print(f"[Memory] 保优策略：旧路径 {old_step_count} 步更优（新 {new_step_count} 步），跳过覆盖：{key}")
                    return
                action = "优化路径" if new_step_count < old_step_count else "同步更新"
                print(f"[Memory] {action}（{old_step_count} → {new_step_count} 步）：{key}")

            # partial → full：质量升级
            if old_quality == "partial" and quality == "full":
                print(f"[Memory] 质量升级（partial → full，{new_step_count} 步）：{key}")

        # ── 构建新记录 ──────────────────────────────────────────────
        # 保留历史使用统计（升级/优化时不清零）
        use_count = existing.get("use_count", 0) if existing else 0
        last_used_at = existing.get("last_used_at") if existing else None
        saved_at = existing.get("saved_at", now) if existing else now

        record = {
            "task": task,
            "quality": quality,
            "step_count": new_step_count,
            "saved_at": saved_at,          # 首次保存时间（不变）
            "updated_at": now,             # 最近更新时间
            "use_count": use_count,        # 保留历史命中次数
        }
        if last_used_at:
            record["last_used_at"] = last_used_at

        if quality == "full":
            record["steps"] = steps
            record["actions"] = actions
        else:
            record["hint"] = hint or {}

        self.flows[key] = record
        self._save()

        if existing is None:
            label = "完全成功（首次）" if quality == "full" else "部分成功（首次）"
        elif existing.get("quality") == "partial" and quality == "full":
            label = f"质量升级 partial→full（{new_step_count} 步）"
        elif quality == "full":
            old_n = existing.get("step_count", "?")
            label = f"完全成功（{old_n}→{new_step_count} 步）"
        else:
            label = "部分成功（更新提示）"
        print(f"[Memory] 已保存任务流程（{label}）：{key}")

    def stats(self) -> dict:
        """返回记忆库的统计摘要，用于调试和监控"""
        total = len(self.flows)
        full_count = sum(1 for v in self.flows.values() if v.get("quality") == "full")
        partial_count = total - full_count
        avg_steps = 0.0
        if full_count > 0:
            avg_steps = sum(
                v.get("step_count", len(v.get("steps", [])))
                for v in self.flows.values() if v.get("quality") == "full"
            ) / full_count
        total_uses = sum(v.get("use_count", 0) for v in self.flows.values())
        return {
            "total": total,
            "full": full_count,
            "partial": partial_count,
            "avg_steps_full": round(avg_steps, 1),
            "total_uses": total_uses,
        }

    def prune(self, max_partial_age_days: int = 30, min_use_for_keep: int = 0) -> int:
        """
        清理低价值记忆，返回删除条数。

        清理规则：
        - partial 记录超过 max_partial_age_days 天未被使用 → 删除
        - use_count == 0 且超过 max_partial_age_days 天的 partial → 删除
        - full 记录不会被自动清理（需手动操作）

        参数：
          max_partial_age_days  partial 记忆最大保留天数（从 updated_at 算起）
          min_use_for_keep      use_count >= 此值的 partial 记录无论多久都保留
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=max_partial_age_days)
        to_delete = []

        for key, data in self.flows.items():
            if data.get("quality") != "partial":
                continue
            use_count = data.get("use_count", 0)
            if use_count >= min_use_for_keep and min_use_for_keep > 0:
                continue
            # 以 updated_at 或 saved_at 判断年龄
            age_str = data.get("updated_at") or data.get("saved_at", "")
            if not age_str:
                to_delete.append(key)
                continue
            try:
                age_dt = datetime.fromisoformat(age_str)
                if age_dt < cutoff:
                    to_delete.append(key)
            except ValueError:
                pass

        for key in to_delete:
            task_preview = self.flows[key].get("task", "")[:40]
            del self.flows[key]
            print(f"[Memory] 清理过期 partial 记忆：{key} | {task_preview}")

        if to_delete:
            self._save()
        print(f"[Memory] prune 完成，删除 {len(to_delete)} 条，剩余 {len(self.flows)} 条")
        return len(to_delete)
