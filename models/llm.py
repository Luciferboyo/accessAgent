import base64
import mimetypes
import os
from openai import OpenAI

# HTTP 请求超时（秒）：连接 + 读取；比 _in_thread 的 150s 略小，确保 SDK 先超时
_HTTP_TIMEOUT = 120.0


class TokenUsage:
    """记录单次调用的 token 用量和费用"""

    def __init__(self, prompt: int = 0, completion: int = 0,
                 model: str = "", price_input: float = 0.0, price_output: float = 0.0):
        self.prompt = prompt
        self.completion = completion
        self.total = prompt + completion
        self.model = model
        self.cost_input = prompt / 1000 * price_input
        self.cost_output = completion / 1000 * price_output
        self.cost = self.cost_input + self.cost_output

    def __str__(self):
        return (f"[{self.model}] "
                f"prompt={self.prompt} completion={self.completion} total={self.total} "
                f"费用=¥{self.cost:.6f}")


def _collect_stream(stream) -> tuple[str, int, int]:
    """
    消费流式响应，返回 (full_content, prompt_tokens, completion_tokens)。
    部分 API 不返回 usage，此时 token 数为 0。
    """
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    print("  ▶ ", end="", flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_content += token
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
    print()  # 换行

    return full_content, prompt_tokens, completion_tokens


class TextLLM:
    """纯文本 LLM，流式输出，实时打印 token"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from config import config
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=_HTTP_TIMEOUT)
        self.model = model
        self.price_input = config.TEXT_PRICE_INPUT
        self.price_output = config.TEXT_PRICE_OUTPUT

    def predict(self, prompt: str, system: str = "") -> tuple[str, TokenUsage]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 优先使用 stream_options 获取 token 用量；
        # 部分 API 不支持该参数会抛异常，自动降级为不带用量统计的普通流式请求
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            full_content, prompt_tokens, completion_tokens = _collect_stream(stream)
        except Exception as e:
            if "stream_options" in str(e) or "include_usage" in str(e):
                print(f"\n  [提示] 该 API 不支持 stream_options，降级为无用量统计模式")
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                full_content, prompt_tokens, completion_tokens = _collect_stream(stream)
            else:
                raise

        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            model=self.model,
            price_input=self.price_input,
            price_output=self.price_output,
        )
        return full_content, usage


class VisionLLM:
    """视觉 LLM，流式输出，只在必要时调用"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from config import config
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=_HTTP_TIMEOUT)
        self.model = model
        self.price_input = config.VISION_PRICE_INPUT
        self.price_output = config.VISION_PRICE_OUTPUT

    def predict(self, prompt: str, image_path: str, system: str = "") -> tuple[str, TokenUsage]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"截图文件不存在，无法调用视觉模型：{image_path}")

        # 自动推断 MIME 类型，兼容 jpg/png/webp 等格式
        mime, _ = mimetypes.guess_type(image_path)
        if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                },
            ],
        })

        # 同上：优先带用量统计，失败则降级
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            full_content, prompt_tokens, completion_tokens = _collect_stream(stream)
        except Exception as e:
            if "stream_options" in str(e) or "include_usage" in str(e):
                print(f"\n  [提示] 该 API 不支持 stream_options，降级为无用量统计模式")
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                full_content, prompt_tokens, completion_tokens = _collect_stream(stream)
            else:
                raise

        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            model=self.model,
            price_input=self.price_input,
            price_output=self.price_output,
        )
        return full_content, usage
