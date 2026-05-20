"""
经验池（Experience Pool）
========================
存储应用/场景特定的操作经验，在执行时按上下文动态检索并注入提示词。

设计目标：
- System prompt 只保留通用规则（始终适用），保持精简且可被 LLM 缓存
- 场景特定规则（如 TG 转发、聊天气泡长按）放进经验池，按需检索
- 检索结果注入 user message（每步动态），不污染 system prompt
- 经验条目可通过编辑 experiences.json 手动扩充，无需改代码
"""

import json
import os
import tempfile


EXPERIENCE_FILE = os.path.join(os.path.dirname(__file__), "experiences.json")


class ExperiencePool:
    def __init__(self, pool_path: str = EXPERIENCE_FILE):
        self.pool_path = pool_path
        self._experiences: list[dict] = self._load()

    # ── 持久化 ──────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if os.path.exists(self.pool_path):
            try:
                with open(self.pool_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[ExperiencePool] 读取失败（{e}），使用内置默认经验")
        data = _default_experiences()
        self._save(data)
        return data

    def _save(self, data: list[dict] = None) -> None:
        if data is None:
            data = self._experiences
        os.makedirs(os.path.dirname(os.path.abspath(self.pool_path)), exist_ok=True)
        try:
            dir_name = os.path.dirname(os.path.abspath(self.pool_path))
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.pool_path)
        except OSError as e:
            print(f"[ExperiencePool] 保存失败（{e}）")

    # ── 检索 ──────────────────────────────────────────────────────────

    def search(
        self,
        task: str = "",
        app_package: str = "",
        history: list[str] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        根据任务描述、当前前台 app 包名、近期操作历史，检索最相关的经验条目。
        使用简单的 tag 关键词计分（无需 embedding API）：
          - 每个 tag 在上下文字符串中出现一次 +1 分
          - 取 score > 0 的条目，按分排序，返回 top_k
        """
        context = " ".join(filter(None, [
            task.lower(),
            app_package.lower(),
            " ".join((history or [])[-6:]).lower(),
        ]))

        scored: list[tuple[int, dict]] = []
        for exp in self._experiences:
            score = sum(1 for tag in exp.get("tags", []) if tag.lower() in context)
            if score > 0:
                scored.append((score, exp))

        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_k]]

    def format_for_prompt(self, experiences: list[dict]) -> str:
        """将检索结果格式化为可直接拼入 user 消息的文本块"""
        if not experiences:
            return ""
        parts = ["📚 相关操作经验（根据当前任务自动检索，请优先遵守）："]
        for exp in experiences:
            parts.append(f"\n【{exp['title']}】\n{exp['rule'].strip()}")
        return "\n".join(parts)

    # ── 扩充 ──────────────────────────────────────────────────────────

    def add(self, experience: dict) -> None:
        """运行时动态添加新经验（并持久化）"""
        self._experiences.append(experience)
        self._save()
        print(f"[ExperiencePool] 已添加经验：{experience.get('id', '?')}")

    def reload(self) -> None:
        """重新从文件加载（外部编辑 experiences.json 后调用）"""
        self._experiences = self._load()
        print(f"[ExperiencePool] 已重新加载，共 {len(self._experiences)} 条经验")


# ── 默认经验库 ────────────────────────────────────────────────────────────

def _default_experiences() -> list[dict]:
    return [
        {
            "id": "im_forward_contact_select",
            "tags": [
                "telegram", "tg", "forward", "转发", "contact", "联系人",
                "share", "分享", "send to", "选择联系人", "org.telegram"
            ],
            "title": "TG/IM 转发联系人选择界面：防止反复 tap 同一位置死循环",
            "rule": """在 TG/微信/WhatsApp 等应用的转发联系人界面，搜索并点击联系人后：
1. 联系人条目旁（左侧头像圆圈）会出现蓝色勾选标记，表示已选中
2. 屏幕【底部】会同时出现蓝色"Share in N chat(s)"或"发送给 N 人"确认按钮

【选中联系人后的强制规则】：
→ 立即把目光移到截图【底部】寻找蓝色确认发送按钮
→ 一旦看到该按钮，必须 tap 其视觉中心，不要再 tap 联系人条目
→ 再次 tap 已选中的联系人会【取消选中】，造成无限循环

【连续多次 tap 同位置无效时】，按顺序检查：
① 截图底部是否有蓝色 Share/Send 按钮 → 有则立即 tap，完成转发
② 联系人左侧头像圆圈是否已有蓝色勾选 → 有则说明已选中，去找底部按钮
③ 联系人未选中 → 尝试 tap 更靠右的区域（名称文字中心，x 坐标 +80~150px）
④ 若联系人头像在截图中有编号方框，优先用 click(index) 而非 tap(x,y)"""
        },
        {
            "id": "im_confirm_button_tap",
            "tags": [
                "share", "send", "forward", "confirm", "发送", "确认", "转发",
                "微信", "wechat", "telegram", "tg", "whatsapp", "分享"
            ],
            "title": "IM 应用转发/分享确认按钮必须直接 tap(x,y)",
            "rule": """在消息应用（微信/TG/WhatsApp 等）中，选完接收人后的底部确认按钮
（如"Share in N chat(s)"、"发送给 N 人"、"转发"等蓝色大按钮）通常在 WebView 中渲染：
→ click(index) 无法触达这类按钮，不要先尝试 click(index) 再等失败
→ 【必须直接 tap(x,y)】，在截图中找到按钮视觉中心，换算为手机坐标后 tap"""
        },
        {
            "id": "chat_bubble_long_press",
            "tags": [
                "chat", "bubble", "image", "photo", "video", "消息", "图片",
                "气泡", "长按", "long_click", "聊天", "forward", "转发"
            ],
            "title": "聊天消息气泡（图片/视频）必须用 tap(x,y) 长按",
            "rule": """聊天列表中的图片/视频/文件消息气泡是 WebView 渲染，没有独立无障碍节点：
→ long_click(index) 会命中错误元素（如标题栏或输入框），无法触发气泡菜单
→ 【必须用 tap(x,y)】精准点击缩略图/气泡的视觉中心，才能触发长按菜单或进入查看器"""
        },
        {
            "id": "planner_im_image_forward",
            "tags": [
                "telegram", "tg", "wechat", "微信", "whatsapp", "image", "photo",
                "图片", "forward", "转发", "消息", "chat", "聊天"
            ],
            "title": "IM 应用图片查找与转发的规划规则（Planner 用）",
            "rule": """规划 TG/微信/WhatsApp 等 IM 应用的图片转发任务时：

【查找最近图片】最短路径：
→ 直接在聊天底部滚动到最新消息区域，最近图片缩略图就在那里
→ 【严禁】规划"进入 Media/相册 页面"或"聊天内搜索关键词定位图片"——这些路径迂回且易失败

【图片发起转发】两种合法路径（合并为一个步骤，任选其一）：
→ 路径A：tap 图片缩略图 → 进入图片查看器 → 点击查看器内的 Forward 按钮
→ 路径B：long_click 图片缩略图 → 弹出操作菜单 → 点击菜单中的 Forward/转发
→ 两条路径都合法有效，"进入图片查看器"是正常推进，不是失败，应继续点击 Forward

【步骤数量控制】：图片转发任务最多 4 步，"找到图片并进入联系人选择界面"合并为一步

【底部确认按钮】：
→ 联系人选择界面底部的"Share in N chat(s)"按钮在 WebView 中渲染，click(index) 可能无效
→ 步骤中应提示执行器：若按钮无响应，改用 tap(x,y) 点击按钮视觉中心"""
        },
        {
            "id": "reflector_image_viewer",
            "tags": [
                "image", "photo", "图片", "viewer", "查看器", "forward", "转发",
                "tap", "thumbnail", "缩略图", "chat", "聊天", "telegram", "tg",
                "wechat", "微信"
            ],
            "title": "聊天图片查看器：tap 缩略图跳转到查看器是正常推进，不能判为 stuck（Reflector 用）",
            "rule": """在聊天中 tap 图片缩略图后，界面会跳转到图片查看器（通常显示 返回/编辑/Forward/更多，约 4-5 个元素）：
→ 这是完全正常的推进，绝对不能判为 stuck

【判断规则】：
→ 步骤目标包含"找到图片"、"发起转发"、"进入转发界面"时：
  · 图片查看器已打开，但尚未点击 Forward → progress，next_hint 提示点击 Forward 按钮
  · 图片查看器已打开并点击了 Forward，进入联系人选择界面 → done
→ 【严禁】因"界面从聊天变成了图片查看器"就判为 stuck——这是 tap 图片的必然结果"""
        },
    ]
