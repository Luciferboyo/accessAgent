import base64
import os
from openai import OpenAI


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


class TextLLM:
    """纯文本 LLM，流式输出，实时打印 token"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from config import config
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.price_input = config.TEXT_PRICE_INPUT
        self.price_output = config.TEXT_PRICE_OUTPUT

    def predict(self, prompt: str, system: str = "") -> tuple[str, TokenUsage]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

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
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.price_input = config.VISION_PRICE_INPUT
        self.price_output = config.VISION_PRICE_OUTPUT

    def predict(self, prompt: str, image_path: str, system: str = "") -> tuple[str, TokenUsage]:
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

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

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

        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            model=self.model,
            price_input=self.price_input,
            price_output=self.price_output,
        )
        return full_content, usage
