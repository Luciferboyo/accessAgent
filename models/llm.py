import base64
import mimetypes
import os
import time
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, BadRequestError, InternalServerError

# HTTP 请求超时（秒）：连接 + 读取；比 _in_thread 的 150s 略小，确保 SDK 先超时
_HTTP_TIMEOUT = 120.0

# 重试策略
_MAX_RETRIES = 3                # 总尝试次数 = _MAX_RETRIES + 1
_RETRY_BASE_DELAY = 1.5         # 指数退避基础（秒）：1.5, 3, 6...
_RETRY_MAX_DELAY = 20.0         # 单次最长退避

# stream_options 不支持的探测：若 BadRequestError 包含这些关键词则降级
_STREAM_OPT_KEYWORDS = ("stream_options", "include_usage", "stream_option")


class TokenUsage:
    """记录单次调用的 token 用量和费用"""

    def __init__(self, prompt: int = 0, completion: int = 0,
                 model: str = "", price_input: float = 0.0, price_output: float = 0.0,
                 usage_available: bool = True):
        self.prompt = prompt
        self.completion = completion
        self.total = prompt + completion
        self.model = model
        self.cost_input = prompt / 1000 * price_input
        self.cost_output = completion / 1000 * price_output
        self.cost = self.cost_input + self.cost_output
        # API 是否实际返回了 usage（False 表示 cost 为 0 是因为 API 不报，而非真实零成本）
        self.usage_available = usage_available

    def __str__(self):
        note = "" if self.usage_available else "（API 未返回用量）"
        return (f"[{self.model}] "
                f"prompt={self.prompt} completion={self.completion} total={self.total} "
                f"费用=¥{self.cost:.6f}{note}")


def _is_stream_options_error(exc: Exception) -> bool:
    """识别"API 不支持 stream_options"这一特定 BadRequest，避免依赖完整字符串匹配。"""
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc).lower()
    return any(k in msg for k in _STREAM_OPT_KEYWORDS)


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    if isinstance(exc, RateLimitError):
        return True
    # BadRequest / Auth 等永久错误不重试
    return False


def _retry_call(fn, *, label: str):
    """指数退避重试包装。fn 应为无参可调用对象。"""
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == _MAX_RETRIES:
                raise
            delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
            etype = type(e).__name__
            print(f"\n  [重试] {label} 第 {attempt + 1}/{_MAX_RETRIES} 次失败（{etype}: {str(e)[:120]}），"
                  f"{delay:.1f}s 后重试")
            time.sleep(delay)
    # 理论上不会到达
    raise last_exc  # type: ignore[misc]


def _collect_stream(stream) -> tuple[str, int, int, bool]:
    """
    消费流式响应，返回 (full_content, prompt_tokens, completion_tokens, usage_available)。
    部分 API 不返回 usage，此时 token 数为 0，usage_available=False。
    """
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    usage_available = False

    print("  ▶ ", end="", flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_content += token
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            usage_available = True
    print()  # 换行

    return full_content, prompt_tokens, completion_tokens, usage_available


def _stream_predict(client, model: str, messages: list, label: str) -> tuple[str, int, int, bool]:
    """
    带重试 + stream_options 探测降级的统一流式调用。
    返回 (content, prompt_tokens, completion_tokens, usage_available)。
    """
    def _call_with_usage():
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        return _collect_stream(stream)

    def _call_plain():
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        return _collect_stream(stream)

    # 优先尝试带 usage 的版本（含重试）
    try:
        return _retry_call(_call_with_usage, label=label)
    except Exception as e:
        if _is_stream_options_error(e):
            print(f"\n  [提示] 该 API 不支持 stream_options，降级为无用量统计模式")
            return _retry_call(_call_plain, label=f"{label}(no_usage)")
        raise


class TextLLM:
    """纯文本 LLM，流式输出，实时打印 token；prompt cache 友好（SYSTEM 应保持完全静态）"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from config import config
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=_HTTP_TIMEOUT)
        self.model = model
        self.price_input = config.TEXT_PRICE_INPUT
        self.price_output = config.TEXT_PRICE_OUTPUT

    def predict(self, prompt: str, system: str = "") -> tuple[str, TokenUsage]:
        # 注意：messages 列表的前缀必须在每次调用间保持稳定，
        # 多数 OpenAI 兼容 API（OpenAI/DeepSeek/Qwen）会对长前缀自动开启 prompt cache。
        # 因此 SYSTEM 字符串应只包含静态规则，动态内容（日期/界面/历史）放进 user 消息。
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        content, prompt_tokens, completion_tokens, usage_avail = _stream_predict(
            self.client, self.model, messages, label=f"TextLLM({self.model})"
        )
        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            model=self.model,
            price_input=self.price_input,
            price_output=self.price_output,
            usage_available=usage_avail,
        )
        return content, usage


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

        content, prompt_tokens, completion_tokens, usage_avail = _stream_predict(
            self.client, self.model, messages, label=f"VisionLLM({self.model})"
        )
        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            model=self.model,
            price_input=self.price_input,
            price_output=self.price_output,
            usage_available=usage_avail,
        )
        return content, usage
