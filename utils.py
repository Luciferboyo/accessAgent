import json
import re


def extract_json(text: str) -> dict:
    “””
    从 LLM 输出中稳健提取 JSON，兼容常见问题：
    - 中文引号 “” → “
    - markdown 代码块包裹
    - JSON 前后有多余文字
    - 字符串值中的原始换行符（导致 JSONDecodeError）
    “””
    text = text.replace(‘“’, ‘”’).replace(‘”’, ‘”’)   # “ “
    text = text.replace(‘‘’, “’”).replace(‘’’, “’”)   # ‘ ‘
    text = re.sub(r’```(?:json)?\s*’, ‘’, text).replace(‘```’, ‘’)
    start = text.find(‘{‘)
    end = text.rfind(‘}’)
    if start == -1 or end == -1:
        raise ValueError(“未找到 JSON 对象”)
    json_str = text[start:end + 1]

    # 第一次尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 第二次尝试：将字符串值中的原始换行符转义为 \n
    # 原理：JSON 合法字符串中的换行必须是 \n（两字符），原始 \n（一字符）非法
    repaired = json_str.replace(‘\n’, ‘\\n’).replace(‘\r’, ‘\\r’)
    return json.loads(repaired)
