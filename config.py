import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # 文本模型：处理 AccessibilityInfo，便宜且快
    TEXT_API_KEY: str = os.getenv("TEXT_API_KEY", "")
    TEXT_BASE_URL: str = os.getenv("TEXT_BASE_URL", "https://api.deepseek.com/v1")
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "deepseek-chat")

    # 视觉模型：只在关键步骤处理截图时调用
    VISION_API_KEY: str = os.getenv("VISION_API_KEY", "")
    VISION_BASE_URL: str = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")

    # 模型定价（单位：元/千token）
    # DeepSeek-chat 默认价格
    TEXT_PRICE_INPUT: float = float(os.getenv("TEXT_PRICE_INPUT", "0.001"))
    TEXT_PRICE_OUTPUT: float = float(os.getenv("TEXT_PRICE_OUTPUT", "0.002"))
    # Qwen-VL-Plus 默认价格
    VISION_PRICE_INPUT: float = float(os.getenv("VISION_PRICE_INPUT", "0.008"))
    VISION_PRICE_OUTPUT: float = float(os.getenv("VISION_PRICE_OUTPUT", "0.008"))

    # WebSocket 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8765

    # Agent 配置
    MAX_STEPS: int = 30          # 最大总步数
    MAX_RETRIES: int = 3         # 连续失败多少次触发重新规划
    MAX_TOTAL_FAILURES: int = 10 # 累计失败上限，超过直接放弃
    MAX_REPLANS: int = 3         # 最多重新规划次数，超过直接放弃

    # 截图保存目录
    SCREENSHOT_DIR: str = "./screenshots"


config = Config()


def _validate_config():
    missing = []
    if not config.TEXT_API_KEY:
        missing.append("TEXT_API_KEY")
    if not config.VISION_API_KEY:
        missing.append("VISION_API_KEY")
    if missing:
        raise EnvironmentError(
            f"缺少必要环境变量：{', '.join(missing)}。"
            f"请在 .env 文件中配置后重新启动。"
        )


_validate_config()
