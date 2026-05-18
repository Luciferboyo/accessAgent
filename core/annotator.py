import os
from PIL import Image, ImageDraw, ImageFont


class ScreenAnnotator:
    """在截图上画编号框，发给视觉 AI"""

    COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    def annotate(self, image_path: str, ui_elements: list[dict], save_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        for elem in ui_elements:
            idx = elem["index"]
            bounds = elem["bounds"]
            x1, y1, x2, y2 = bounds
            # 坐标修正：确保 x1<=x2, y1<=y2，跳过无效元素
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            color = self.COLORS[idx % len(self.COLORS)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label_bg = [x1, y1 - 28, x1 + 32, y1]
            draw.rectangle(label_bg, fill=color)
            draw.text((x1 + 4, y1 - 26), str(idx), fill="white", font=font)

        img.save(save_path)
        return save_path

    def save_screenshot(self, image_b64: str, directory: str, step: int) -> str:
        import base64
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"step_{step:03d}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return path
