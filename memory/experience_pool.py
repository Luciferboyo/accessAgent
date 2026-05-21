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
        defaults = _default_experiences()
        if os.path.exists(self.pool_path):
            try:
                with open(self.pool_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                # 自动合并：将默认经验中 ID 不存在于文件中的条目追加进去
                existing_ids = {e.get("id") for e in existing}
                new_entries = [d for d in defaults if d.get("id") not in existing_ids]
                if new_entries:
                    merged = existing + new_entries
                    self._save(merged)
                    print(f"[ExperiencePool] 自动合并 {len(new_entries)} 条新经验：{[e['id'] for e in new_entries]}")
                    return merged
                return existing
            except (json.JSONDecodeError, OSError) as e:
                print(f"[ExperiencePool] 读取失败（{e}），使用内置默认经验")
        self._save(defaults)
        return defaults

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
        {
            "id": "tg_username_vs_display_name",
            "tags": [
                "telegram", "tg", "username", "用户名", "display name", "显示名",
                "title", "标题", "verify", "验证", "mismatch", "不匹配", "back", "返回",
                "search", "搜索", "@", "联系人", "contact"
            ],
            "title": "TG的@username与显示名是不同字段，搜索结果不要因显示名不同而拒绝",
            "rule": """Telegram 联系人有两个独立字段：
  - 显示名（Display Name）：用户自定义的昵称，如 "GLO008"、"Benny Lee"
  - 用户名（Username）：@开头的唯一ID，如 "@bgyy008"

【重要】：这两个字段完全可以不同，任务中的名字可能指其中任意一个。

当任务描述的是 @username（如"转发给 @bgyy008"）：
→ 在搜索框输入 @bgyy008 → TG 返回的搜索结果就是该联系人
→ 进入聊天后，标题栏显示的是【显示名】（如 GLO008），不是 @bgyy008
→ 显示名 ≠ @username 完全正常，不是错误，不应该 back

【验证 @username 的正确方式】（仅在怀疑时）：
→ tap 聊天标题栏 → 进入联系人资料页 → 查看资料页中的 "@xxx" 用户名字段
→ 若资料页 @username 与任务吻合 → 正确联系人，back 返回聊天继续任务

【绝对禁止】：
✗ 用显示名（GLO008）与任务描述的 @username（@bgyy008）直接比较，因"不一样"就 back
✗ 一旦发现"标题不完全等于任务名字"就立刻 back，这会导致正确联系人被反复拒绝"""
        },
        {
            "id": "tg_search_username_at_prefix",
            "tags": [
                "telegram", "tg", "search", "搜索", "username", "用户名",
                "找", "find", "contact", "联系人", "@", "bgyy", "gl008"
            ],
            "title": "TG搜索联系人必须带@前缀，否则本地聊天列表搜不到",
            "rule": """在 Telegram 搜索框中查找联系人时，必须使用 @ 前缀输入完整用户名：

【正确做法】：输入 @bgyy008 → 精确命中该用户名
【错误做法】：输入 bgyy008 → 仅搜索本地聊天历史，可能返回 No results
【错误做法】：输入 GL008 → 按显示名搜索，会命中大量相似名称（GLO008、GL002 等）

【强制规则】：
① type 操作的 text 字段必须以 @ 开头，如 "@bgyy008"
② 输入前确认搜索框已清空（若框内有内容先点击 Clear 按钮再输入）
③ 搜索框内只允许有一个关键词，不要追加内容到已有文字后面
④ 若 @username 搜索仍无结果，说明该用户从未与当前账号聊过天，需要通过"New Message"功能全局搜索"""
        },
        {
            "id": "tg_chat_title_verify",
            "tags": [
                "telegram", "tg", "chat", "聊天", "enter", "进入", "contact", "联系人",
                "username", "用户名", "forward", "转发", "title", "标题"
            ],
            "title": "进入TG聊天后验证标题栏——区分「@username搜索」与「列表滚动」两种入口",
            "rule": """进入 TG 聊天后如何验证是否进入了正确的聊天，取决于【如何找到这个联系人】：

━━ 情况A：通过搜索 @username 进入（最常见） ━━
→ 搜索框输入 @bgyy008 → TG 全局搜索直接返回该用户 → 点击结果进入聊天
→ 【此时标题栏显示的是"显示名"（如 GLO008、Benny 等），而非 @username】
→ 显示名与任务中的 @username 不同是完全正常的，两者是同一个人的不同称呼
→ 【强制规则】：通过 @username 搜索进入的聊天，直接信任，不要因显示名不同而 back
→ 若需确认，tap 标题栏进入联系人资料页，查看资料页上的 @username 字段是否吻合

━━ 情况B：从聊天列表滚动查找后进入 ━━
→ 列表中可能有 GL002/GL005/GL008 等相似名称，视觉模型容易选错
→ 进入后截图确认标题栏显示的名称是否与任务目标一致
→ 不匹配 → 立即 back，重新仔细选择正确的联系人

━━ 判断自己属于哪种情况 ━━
→ 上一步执行了 type "@xxx" + 搜索框输入 → 情况A，信任结果，继续任务
→ 上一步是在聊天列表滚动后点击 → 情况B，验证标题栏

【反模式（禁止）】：
✗ 通过 @username 搜索进入聊天后，因标题显示名与任务描述的名字不完全相同就 back
✗ 反复 back → 重搜 → 进入同一个联系人 → 又 back，造成无限循环"""
        },
        {
            "id": "tg_scroll_safe_zone",
            "tags": [
                "telegram", "tg", "chat", "聊天", "scroll", "滚动", "图片", "image",
                "photo", "消息", "message", "查找", "find"
            ],
            "title": "TG聊天内scroll必须在屏幕中部执行，顶部滑动会跳转到联系人资料页",
            "rule": """在 TG 聊天界面查找消息/图片时，scroll 操作有严格区域限制：

【危险区域（禁止 scroll）】：y < 屏幕高度 15%（以 2340px 屏幕为例，约 y < 350）
  → 这是聊天标题栏/联系人头像区域，在此处滑动会触发"进入联系人资料页"导航

【安全区域（推荐 scroll）】：y 在屏幕高度 30%-80% 之间（约 700-1870px）
  → 这是聊天消息列表区域，滑动才能真正滚动消息

【强制规则】：
① 选择用于 scroll 的元素时，只选 y 坐标在屏幕中部的元素（聊天消息列表）
② 若无中部可 scroll 元素，改用 tap(x, y_mid) 在消息区中部位置直接滑动（y_mid ≈ 屏幕高度 50%）
③ 绝对不要用 index 对应 y < 350 的元素执行 scroll"""
        },
    ]
