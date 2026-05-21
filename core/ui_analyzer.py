from typing import Any


class UIAnalyzer:
    """
    分析无障碍树，判断当前步骤是否需要截图。
    能从文字信息决策的就不调用视觉模型。
    """

    # 触发视觉模式的文本关键词：验证码、滑块、拼图、人机识别、点选等
    # 这些场景必须用截图，文字信息完全不足以决策
    NEEDS_VISION_KEYWORDS = [
        # 验证码类
        "验证码", "captcha", "recaptcha",
        # 图形识别 / 点选验证
        "图片", "图形", "选择图中", "请点击", "点选",
        # 滑块 / 拼图验证
        "滑动验证", "滑块", "拖动", "拼图", "puzzle", "slider", "drag",
        # 人机验证
        "人机验证", "verify you are human", "i'm not a robot",
    ]

    # 当 UI 树中 WebView 占主导（>50% 元素）时，强制走视觉
    # 因为 WebView 内部内容不在无障碍树里，文字描述全是空 WebView 容器
    WEBVIEW_CLASS_HINTS = ("WebView", "webview")

    @staticmethod
    def _valid_bounds(elem: dict) -> bool:
        """判断元素边界是否有效（宽和高均 >= 2 像素）"""
        b = elem.get("bounds", [0, 0, 0, 0])
        return abs(b[2] - b[0]) >= 2 and abs(b[3] - b[1]) >= 2

    def parse_elements(self, ui_elements: list[dict]) -> str:
        """
        将元素列表转成文字描述，发给文本 LLM。
        自动跳过 bounds 无效的不可见元素（与 ScreenAnnotator 保持一致），
        避免 LLM 选取无法点击的零尺寸元素导致死循环。
        """
        lines = []
        for elem in ui_elements:
            if not self._valid_bounds(elem):
                continue  # 跳过不可见元素，与标注器保持一致

            parts = [f"[{elem['index']}]", elem["class"]]
            if elem.get("text"):
                parts.append(f'text="{elem["text"]}"')
            if elem.get("content_desc"):
                parts.append(f'desc="{elem["content_desc"]}"')
            if elem.get("resource_id"):
                parts.append(f'id="{elem["resource_id"]}"')

            flags = []
            if elem.get("clickable"):
                flags.append("可点击")
            if elem.get("editable"):
                flags.append("可输入")
            if elem.get("scrollable"):
                flags.append("可滚动")
            if flags:
                parts.append(f'({"/".join(flags)})')

            lines.append(" ".join(parts))
        return "\n".join(lines)

    def needs_screenshot(self, ui_elements: list[dict], last_action_result: str = "") -> bool:
        """
        判断当前步骤是否需要截图：
        - 存在验证码/滑块/人机验证类关键词
        - 可见元素中普遍没有有效 text/desc（纯图形界面）
        - WebView 占主导（无障碍树看不到 WebView 内部内容）
        - 上一步执行结果不确定

        注意：仅对 bounds 有效的可见元素做比例判断，
        避免大量零尺寸虚拟节点拉低比例，导致误触发截图。
        """
        # 1) 关键词命中：合并 text/desc 后做单次 in 判断（O(N×K) → O(N+K)）
        kw_lower = tuple(kw.lower() for kw in self.NEEDS_VISION_KEYWORDS)
        for elem in ui_elements:
            blob = (elem.get("text", "") + " " + elem.get("content_desc", "")).lower()
            if any(kw in blob for kw in kw_lower):
                return True

        # 只统计可见元素
        visible = [e for e in ui_elements if self._valid_bounds(e)]
        if not visible:
            return True

        # 2) WebView 占主导：WebView 容器在无障碍树里就是空盒子，必须看截图
        webview_count = sum(
            1 for e in visible
            if any(hint in e.get("class", "") for hint in self.WEBVIEW_CLASS_HINTS)
        )
        if webview_count > 0 and webview_count >= len(visible) * 0.5:
            return True

        # 3) 文字信息稀疏：可解读元素 < 30%
        meaningful = [
            e for e in visible
            if e.get("text") or e.get("content_desc") or e.get("resource_id")
        ]
        if len(meaningful) < len(visible) * 0.3:
            return True

        return False

    def find_element_by_text(self, ui_elements: list[dict], text: str) -> dict | None:
        """通过文字直接定位元素，不需要 AI"""
        for elem in ui_elements:
            if text in elem.get("text", "") or text in elem.get("content_desc", ""):
                return elem
        return None

    def get_center(self, elem: dict) -> tuple[int, int]:
        """计算元素中心坐标"""
        bounds = elem["bounds"]
        x = (bounds[0] + bounds[2]) // 2
        y = (bounds[1] + bounds[3]) // 2
        return x, y
