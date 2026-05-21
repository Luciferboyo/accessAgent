"""
中国法定节假日检测（含调休补班日识别）
======================================
基于 timor.tech 免费 API，结果按日期缓存到进程内存 + JSON 文件。

返回值：
- "workday" 普通工作日
- "weekend" 普通周末
- "holiday" 法定节假日（春节、国庆等）
- "makeup"  调休补班日（如：周六上班为周一调休补班）
- "unknown" API 失败/网络异常，调用方需决定如何处理
"""

import json
import os
import tempfile
from datetime import date

import httpx


_CACHE_FILE = "./memory/holiday_cache.json"
_API_TIMEOUT = 5.0


def _load_cache() -> dict[str, str]:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    dir_name = os.path.dirname(os.path.abspath(_CACHE_FILE)) or "."
    os.makedirs(dir_name, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CACHE_FILE)
    except OSError:
        pass


# 模块级缓存：避免每次触发都读文件
_cache: dict[str, str] = _load_cache()

# timor.tech 返回 type.type 编码 → 我们的语义
# 0 工作日 / 1 休息日(周末) / 2 节假日 / 3 调休补班
_TYPE_MAP = {0: "workday", 1: "weekend", 2: "holiday", 3: "makeup"}


async def get_day_type(d: date | None = None) -> str:
    """
    查询某天的日期类型。默认查询今天。
    成功结果会持久化缓存；API 失败返回 "unknown"，不写入缓存（下次重试）。
    """
    if d is None:
        d = date.today()
    key = d.isoformat()

    if key in _cache:
        return _cache[key]

    url = f"https://timor.tech/api/holiday/info/{key}"
    try:
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.get(url)
            data = resp.json()
            if data.get("code") != 0:
                return "unknown"
            t_block = data.get("type") or {}
            code = t_block.get("type", -1)
            result = _TYPE_MAP.get(code, "unknown")
            if result != "unknown":
                _cache[key] = result
                _save_cache(_cache)
            return result
    except Exception as e:
        print(f"[Holiday] {key} 查询失败（{type(e).__name__}: {str(e)[:80]}），返回 unknown")
        return "unknown"


async def is_workday(d: date | None = None, treat_makeup_as_workday: bool = True) -> bool:
    """
    便捷判断：今天是否应该按工作日执行。
    - workday / makeup（默认）→ True
    - weekend / holiday → False
    - unknown → True（兜底：API 不通时默认执行，避免错过打卡）
    """
    t = await get_day_type(d)
    if t == "workday":
        return True
    if t == "makeup":
        return treat_makeup_as_workday
    if t in ("weekend", "holiday"):
        return False
    return True   # unknown 兜底
