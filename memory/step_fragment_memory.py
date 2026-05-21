"""
步骤片段记忆库（Step Fragment Memory）
=======================================
粒度比 TaskMemory 更细：将成功任务的每个步骤单独存储，
支持跨任务的步骤级经验复用。

与其他记忆层的分工：
  ExperiencePool       → 人工编写的通用规则，始终注入
  TaskMemory           → 任务粒度，高相似时直接复用完整计划
  StepFragmentMemory   → 步骤粒度，跨任务积累场景经验，注入 Planner 提示

核心思想：
  - "打开TG→搜索联系人" 在"发消息"和"转发图片"两个任务里完全相同，应当共享
  - 不同任务的相同子场景积累越多，Planner 的规划越准确
  - 参数（用户名/内容等）不存入片段，只存操作意图，避免参数污染
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime


FRAGMENT_FILE = "./memory/step_fragments.json"


class StepFragmentMemory:

    def __init__(self, fragment_path: str = FRAGMENT_FILE):
        self.fragment_path = fragment_path
        os.makedirs(os.path.dirname(os.path.abspath(fragment_path)), exist_ok=True)
        self.fragments: list[dict] = self._load()

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if os.path.exists(self.fragment_path):
            try:
                with open(self.fragment_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[FragmentMemory] 读取失败（{e}），重置为空")
        return []

    def _save(self) -> None:
        dir_name = os.path.dirname(os.path.abspath(self.fragment_path))
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.fragments, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.fragment_path)
        except OSError as e:
            print(f"[FragmentMemory] 保存失败（{e}）")

    # ── 关键词提取 & 相似度 ───────────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> str:
        """
        预处理：去掉干扰字符再分词，提升中文 n-gram 覆盖率。
        - 去掉 @username（具体参数，不参与匹配）
        - 去掉数字坐标（如 1080 2340）
        - 将常见分隔符替换成空格，让中文连续段更完整
        """
        import re
        text = re.sub(r'@\w+', '', text)          # 去掉 @用户名
        text = re.sub(r'\b\d{3,}\b', '', text)    # 去掉长数字
        text = re.sub(r'[_\-/\\|→·•]', ' ', text) # 分隔符转空格
        return text.strip()

    def _extract_keywords(self, text: str) -> set[str]:
        """提取中英文关键词（预处理后再提取，提升中文覆盖率）"""
        text = self._preprocess(text)
        keywords = set()
        for word in text.split():
            w = word.lower().strip('.,!?，。！？()（）[]【】')
            if len(w) >= 2:
                keywords.add(w)
        chinese_chars = ""
        for ch in text:
            if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
                chinese_chars += ch
            else:
                if len(chinese_chars) >= 2:
                    for l in range(2, min(5, len(chinese_chars) + 1)):
                        for i in range(len(chinese_chars) - l + 1):
                            keywords.add(chinese_chars[i:i + l])
                chinese_chars = ""
        if len(chinese_chars) >= 2:
            for l in range(2, min(5, len(chinese_chars) + 1)):
                for i in range(len(chinese_chars) - l + 1):
                    keywords.add(chinese_chars[i:i + l])
        return keywords

    def _similarity(self, query: str, doc: str) -> float:
        """
        混合相似度：0.4×Jaccard + 0.6×查询召回率。

        纯 Jaccard 对长文档不友好（分母大导致分数低）；
        召回率 = 查询关键词中有多少出现在文档中，对短查询更敏感。
        两者混合后，既保留整体相似性，又对关键词命中更敏感。
        """
        kw1 = self._extract_keywords(query)
        kw2 = self._extract_keywords(doc)
        if not kw1 or not kw2:
            return 0.0
        intersection = len(kw1 & kw2)
        if intersection == 0:
            return 0.0
        jaccard = intersection / len(kw1 | kw2)
        recall = intersection / len(kw1)   # 查询关键词被文档覆盖的比例
        return 0.4 * jaccard + 0.6 * recall

    # ── 动作摘要（去参数化）────────────────────────────────────────────────

    @staticmethod
    def _summarize_action(action: dict) -> str:
        """
        将单个动作转为去参数化的意图描述。
        参数值（具体用户名/文本内容/坐标等）不存入，只保留操作类型和意图。
        这样片段可跨任务复用，不被具体参数污染。
        """
        act = action.get("action", "")
        reason = action.get("reason", "")

        # 意图优先（取 reason 前 40 字，通常已包含操作意图）
        if reason:
            # 截掉具体参数：把引号内容、@xxx、数字坐标替换为占位符
            import re
            cleaned = re.sub(r'"[^"]{1,30}"', '"<内容>"', reason)
            cleaned = re.sub(r'@\w+', '@<用户名>', cleaned)
            cleaned = re.sub(r'\b\d{3,}\b', '<坐标>', cleaned)
            return cleaned[:50]

        # 兜底：用动作类型描述
        type_labels = {
            "click": "点击元素", "long_click": "长按元素",
            "type": "输入文字", "scroll": "滚动列表",
            "tap": "精准点击", "back": "返回上一页",
            "open_app": "打开应用", "find_package": "查询包名",
            "search_web": "搜索网页", "finish": "完成任务",
            "step_done": "确认步骤完成",
        }
        return type_labels.get(act, act)

    # ── 核心：从成功任务中提取片段 ─────────────────────────────────────────

    def ingest_task(self, task: str, app: str,
                    plan: list[str], action_log: list[dict]) -> int:
        """
        从成功完成的任务中提取步骤片段并保存。

        参数：
          task        原始任务描述
          app         任务主要使用的 App 包名（如 org.telegram.messenger）
          plan        计划步骤列表
          action_log  执行日志（每条应含 _step_index / _step_goal 字段）

        返回：新写入或更新的片段数
        """
        if not plan:
            return 0

        # 按 _step_index 分组 actions
        step_actions: dict[int, list[dict]] = {}
        for a in action_log:
            idx = a.get("_step_index", 0)
            step_actions.setdefault(idx, []).append(a)

        # 若 action_log 没有步骤标注（旧格式），均匀分配
        if not any("_step_index" in a for a in action_log):
            n = len(plan)
            m = len(action_log)
            for i in range(n):
                step_actions[i] = action_log[i * m // n: (i + 1) * m // n]

        saved = 0
        for i, step_goal in enumerate(plan):
            actions_for_step = step_actions.get(i, [])
            # 去掉内部标注字段，只保留真实动作字段
            clean_actions = [
                {k: v for k, v in a.items() if not k.startswith("_")}
                for a in actions_for_step
            ]
            summaries = [self._summarize_action(a) for a in clean_actions]
            action_summary = " → ".join(s for s in summaries if s)

            updated = self._upsert(
                app=app,
                step_goal=step_goal,
                action_summary=action_summary,
                source_task=task,
            )
            if updated:
                saved += 1

        if saved > 0:
            self._save()
            print(f"[FragmentMemory] 已更新 {saved} 个步骤片段（来自任务：{task[:40]}）")
        return saved

    def _upsert(self, app: str, step_goal: str,
                action_summary: str, source_task: str) -> bool:
        """
        插入或更新片段。
        相同 app + 相似 goal（Jaccard > 0.6）视为同一场景，合并统计。
        """
        MERGE_THRESHOLD = 0.6
        now = datetime.now().isoformat()

        for frag in self.fragments:
            if frag.get("app") != app:
                continue
            sim = self._similarity(step_goal, frag["step_goal"])
            if sim >= MERGE_THRESHOLD:
                # 合并：累加成功次数，优先保留更长（更详细）的摘要
                frag["success_count"] = frag.get("success_count", 1) + 1
                frag["updated_at"] = now
                if action_summary and len(action_summary) > len(frag.get("action_summary", "")):
                    frag["action_summary"] = action_summary
                # 记录来源任务（去重，最多保留 5 个）
                sources = frag.setdefault("source_tasks", [])
                task_preview = source_task[:50]
                if task_preview not in sources:
                    sources.append(task_preview)
                    if len(sources) > 5:
                        sources.pop(0)
                return True

        # 新片段
        frag_id = hashlib.sha256(f"{app}|{step_goal}".encode()).hexdigest()[:12]
        self.fragments.append({
            "id": frag_id,
            "app": app,
            "step_goal": step_goal,
            "action_summary": action_summary,
            "success_count": 1,
            "source_tasks": [source_task[:50]],
            "created_at": now,
            "updated_at": now,
        })
        return True

    # ── 检索 ──────────────────────────────────────────────────────────────

    def find_relevant(self, query: str, app: str = "",
                      top_k: int = 5, min_score: float = 0.08) -> list[dict]:
        """
        双轨检索：结合任务级相似度 + 步骤级相似度，避免词汇不重叠导致漏检。

        轨道 1（任务级）：query vs 片段的来源任务（source_tasks）
          - 原理：如果历史任务与当前任务相似，其所有子步骤都可能有参考价值
          - 优势：任务描述比步骤目标更口语化，词汇更接近查询

        轨道 2（步骤级）：query vs 片段的步骤目标（step_goal）
          - 原理：跨任务的相同子场景（如"打开应用"出现在所有任务）
          - 优势：覆盖来源任务不同但子步骤相同的情况

        最终分数 = max(任务级, 步骤级) × app权重 × 成功次数加权
        LLM（Planner）再做最终语义筛选，无需在此过滤太严格。
        """
        # 先缓存所有来源任务的相似度（避免重复计算）
        source_sim_cache: dict[str, float] = {}
        for frag in self.fragments:
            for src in frag.get("source_tasks", []):
                if src not in source_sim_cache:
                    source_sim_cache[src] = self._similarity(query, src)

        scored: list[tuple[float, dict]] = []
        for frag in self.fragments:
            frag_app = frag.get("app", "")
            app_factor = 1.0 if (not app or not frag_app or frag_app == app) else 0.4

            # 轨道1：来源任务相似度（取最高分来源）
            task_sim = max(
                (source_sim_cache.get(s, 0.0) for s in frag.get("source_tasks", [])),
                default=0.0
            )

            # 轨道2：步骤目标相似度
            step_sim = self._similarity(query, frag["step_goal"])

            # 成功次数加权（验证越多越可信，最高 +25%）
            confidence_bonus = min(0.25, frag.get("success_count", 1) * 0.05)

            # 取双轨最高分（任务级权重稍高，因为词汇更接近）
            sim = max(task_sim * 1.1, step_sim) * app_factor
            score = sim * (1 + confidence_bonus)

            if score >= min_score:
                scored.append((score, frag))

        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:top_k]]

    def format_for_planner(self, fragments: list[dict]) -> str:
        """
        将检索到的片段格式化为 Planner 可用的提示文本。
        只提供"做过什么、怎么做的"，不复制具体参数。
        """
        if not fragments:
            return ""
        lines = ["📋 历史步骤经验（仅供规划参考，参数请根据当前任务调整）："]
        for i, frag in enumerate(fragments, 1):
            app_label = frag.get("app", "").split(".")[-1] or "通用"
            sc = frag.get("success_count", 1)
            goal = frag["step_goal"][:60]
            summary = frag.get("action_summary", "")[:80]
            lines.append(
                f"  {i}. 场景：[{app_label}] {goal}\n"
                f"     执行路径：{summary}\n"
                f"     历史验证：{sc} 次成功"
            )
        return "\n".join(lines)

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回片段库统计摘要"""
        total = len(self.fragments)
        by_app: dict[str, int] = {}
        for frag in self.fragments:
            app = frag.get("app", "unknown").split(".")[-1]
            by_app[app] = by_app.get(app, 0) + 1
        avg_success = (
            sum(f.get("success_count", 1) for f in self.fragments) / total
            if total else 0
        )
        return {
            "total_fragments": total,
            "by_app": by_app,
            "avg_success_count": round(avg_success, 1),
        }

    def prune(self, min_success: int = 1) -> int:
        """删除成功次数低于阈值的片段（清理噪声数据）"""
        before = len(self.fragments)
        self.fragments = [
            f for f in self.fragments
            if f.get("success_count", 1) >= min_success
        ]
        removed = before - len(self.fragments)
        if removed:
            self._save()
            print(f"[FragmentMemory] prune 删除 {removed} 条低质量片段")
        return removed
