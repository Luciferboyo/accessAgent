import json
import re


def extract_json(text: str) -> dict:
    """
    从 LLM 输出中稳健提取 JSON，兼容常见问题：
    - 中文引号 "" → "
    - markdown 代码块包裹
    - JSON 前后有多余文字
    """
    text = text.replace('“', '"').replace('”', '"')   # " "
    text = text.replace('‘', "'").replace('’', "'")   # ' '
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("未找到 JSON 对象")
    return json.loads(text[start:end + 1])
