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


def _try_parse(json_str: str) -> dict:
    """先直接解析，失败则尝试修复字符串内换行符再解析。"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    repaired = _escape_string_newlines(json_str)
    return json.loads(repaired)


def extract_json(text: str) -> dict:
    """
    从 LLM 输出中稳健提取 JSON，兼容常见问题：
    - markdown 代码块包裹
    - JSON 前后有多余文字
    - 字符串值中的原始换行符
    - 中文弯引号的两种用法：
        1. 作为 JSON 字符串定界符（替换为直引号）
        2. 作为行内引号嵌套在字符串值中（移除，避免破坏 JSON 结构）
    """
    # 先去掉 markdown 代码块
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
    # 中文单引号统一替换
    text = text.replace('‘', "'").replace('’', "'")

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("未找到 JSON 对象")
    json_str = text[start:end + 1]

    # 策略 1：中文双弯引号作为 JSON 定界符 → 替换为直引号
    s1 = json_str.replace('“', '"').replace('”', '"')
    try:
        return _try_parse(s1)
    except json.JSONDecodeError:
        pass

    # 策略 2：中文双弯引号作为行内引号（嵌套在字符串值里）→ 直接移除
    s2 = json_str.replace('“', '').replace('”', '')
    try:
        return _try_parse(s2)
    except json.JSONDecodeError:
        pass

    # 策略 3：中文双弯引号替换为转义引号 \"（另一种嵌套场景）
    s3 = json_str.replace('“', '\\"').replace('”', '\\"')
    try:
        return _try_parse(s3)
    except json.JSONDecodeError:
        pass

    # 所有策略均失败，抛出最后一个错误
    raise json.JSONDecodeError("所有解析策略均失败", json_str, 0)
