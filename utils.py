import json
import re


def _escape_string_newlines(s: str) -> str:
    """
    只转义 JSON 字符串值内部的原始换行符，不影响结构性换行。
    逐字符扫描，追踪是否处于引号内。
    """
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == '\\':
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        else:
            result.append(ch)
    return ''.join(result)


def extract_json(text: str) -> dict:
    """
    从 LLM 输出中稳健提取 JSON，兼容常见问题：
    - 中文引号转英文引号
    - markdown 代码块包裹
    - JSON 前后有多余文字
    - 字符串值中的原始换行符（导致 JSONDecodeError）
    """
    text = text.replace('“', '"').replace('”', '"')   # 中文双引号
    text = text.replace('‘', "'").replace('’', "'")   # 中文单引号
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("未找到 JSON 对象")
    json_str = text[start:end + 1]

    # 第一次尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 第二次尝试：只转义字符串值内部的原始换行符，保留结构性换行
    repaired = _escape_string_newlines(json_str)
    return json.loads(repaired)
