import os
from PIL import Image, ImageDraw, ImageFont


class ScreenAnnotator:
    """在截图上画编号框，发给视觉 AI"""

    COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    def annotate(self, image_path: str, ui_elements: list[dict], save_path: str,
                 screen_size: list[int] = None) -> str:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        # 计算手机坐标系 → 图片坐标系的缩放比例
        # 手机截图经过压缩（max_width=720），但 UI 元素 bounds 仍是手机原始坐标
        if screen_size and screen_size[0] > 0 and screen_size[1] > 0:
            scale_x = img_w / screen_size[0]
            scale_y = img_h / screen_size[1]
        else:
            # 兜底：从元素坐标范围推断
            xs = [e["bounds"][2] for e in ui_elements if e.get("bounds")]
            ys = [e["bounds"][3] for e in ui_elements if e.get("bounds")]
            max_x = max(xs) if xs else img_w
            max_y = max(ys) if ys else img_h
            scale_x = img_w / max_x if max_x > img_w else 1.0
            scale_y = img_h / max_y if max_y > img_h else 1.0

        draw = ImageDraw.Draw(img)

        # 按优先级尝试常见字体；全部失败时降级为 PIL 内置默认字体
        _FONT_CANDIDATES = [
            "arial.ttf",
            "/system/fonts/Roboto-Regular.ttf",   # Android
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",          # Windows 绝对路径
        ]
        font = None
        for _fc in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(_fc, 24)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        for elem in ui_elements:
            idx = elem["index"]
            bounds = elem["bounds"]
            # 将手机坐标系缩放到图片坐标系
            x1 = int(bounds[0] * scale_x)
            y1 = int(bounds[1] * scale_y)
            x2 = int(bounds[2] * scale_x)
            y2 = int(bounds[3] * scale_y)
            # 坐标修正：确保 x1<=x2, y1<=y2，跳过无效元素
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            color = self.COLORS[idx % len(self.COLORS)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 标签背景：y 不能为负，贴近顶部时往下偏移
            label_top = max(0, y1 - 28)
            label_bottom = label_top + 28
            label_bg = [x1, label_top, x1 + 32, label_bottom]
            draw.rectangle(label_bg, fill=color)
            draw.text((x1 + 4, label_top + 2), str(idx), fill="white", font=font)

        img.save(save_path)
        return save_path

    def save_screenshot(self, image_b64: str, directory: str, step: int) -> str:
        import base64
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"step_{step:03d}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return path
