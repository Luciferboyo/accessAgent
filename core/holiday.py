"""
节假日检测（可配置 provider + 地区）
====================================
内置两个 provider，可按 schedule 配置选择：

- "china_timor"（默认）：中国法定节假日，使用 timor.tech 免费 API
  - 识别工作日 / 周末 / 法定节假日 / 调休补班
  - region 字段无效（永远是中国大陆）

- "nager"：国际通用，使用 https://date.nager.at（免费、覆盖 100+ 国家）
  - region 必须填 ISO 国家码（如 US/JP/GB/DE/SG/MY 等）
  - 周末本地判定（周六/日），节假日按 nager 返回；不区分调休补班

外部可调用 `get_day_type(date, provider, region)` 统一获取语义化结果。
失败兜底返回 "unknown"，由调用方决定如何处理（默认按工作日执行，避免错过打卡）。
"""

import json
import os
import tempfile
from datetime import date

import httpx


_CACHE_FILE = "./memory/holiday_cache.json"
_API_TIMEOUT = 5.0


# ── 文件级缓存 ────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    dir_name = os.path.dirname(os.path.abspath(_CACHE_FILE)) or "."
    os.makedirs(dir_name, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CACHE_FILE)
    except OSError:
        pass


# 缓存结构（含两种 provider）：
# {
#   "timor": {"2026-05-22": "workday", ...},
#   "nager:US": {"2026": ["2026-01-01", "2026-07-04", ...]},
# }
_cache: dict = _load_cache()


# ── timor.tech（中国） ────────────────────────────────────────────────

_TIMOR_TYPE_MAP = {0: "workday", 1: "weekend", 2: "holiday", 3: "makeup"}


async def _query_timor(d: date) -> str:
    bucket = _cache.setdefault("timor", {})
    key = d.isoformat()
    if key in bucket:
        return bucket[key]

    url = f"https://timor.tech/api/holiday/info/{key}"
    try:
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.get(url)
            data = resp.json()
            if data.get("code") != 0:
                return "unknown"
            t_block = data.get("type") or {}
            code = t_block.get("type", -1)
            result = _TIMOR_TYPE_MAP.get(code, "unknown")
            if result != "unknown":
                bucket[key] = result
                _save_cache(_cache)
            return result
    except Exception as e:
        print(f"[Holiday/timor] {key} 查询失败（{type(e).__name__}: {str(e)[:80]}）")
        return "unknown"


# ── nager.date（国际通用） ────────────────────────────────────────────

async def _fetch_nager_year(year: int, country: str) -> list[str] | None:
    """获取某国某年的全部公共节假日（一次拉一年，本地比对）。"""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
    try:
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return [item["date"] for item in data if "date" in item]
    except Exception as e:
        print(f"[Holiday/nager] {country} {year} 查询失败（{type(e).__name__}: {str(e)[:80]}）")
        return None


async def _query_nager(d: date, country: str) -> str:
    if not country:
        return "unknown"
    country = country.upper()
    bucket_key = f"nager:{country}"
    bucket = _cache.setdefault(bucket_key, {})
    year_key = str(d.year)

    # 命中年度缓存：直接本地比对
    if year_key not in bucket:
        holidays = await _fetch_nager_year(d.year, country)
        if holidays is None:
            return "unknown"
        bucket[year_key] = holidays
        _save_cache(_cache)

    holiday_list = bucket[year_key]
    date_str = d.isoformat()
    if date_str in holiday_list:
        return "holiday"
    # 周末本地判定
    if d.weekday() >= 5:
        return "weekend"
    return "workday"


# ── 统一入口 ──────────────────────────────────────────────────────────

PROVIDERS = {
    "china_timor": _query_timor,
    "nager":       _query_nager,
}


async def get_day_type(d: date | None = None,
                       provider: str = "china_timor",
                       region: str = "") -> str:
    """
    返回 'workday' / 'weekend' / 'holiday' / 'makeup' / 'unknown'。
    provider:
      - 'china_timor': 中国大陆，region 字段忽略
      - 'nager':       国际，region 必填 ISO 国家码（US/JP/GB 等）
    """
    if d is None:
        d = date.today()
    fn = PROVIDERS.get(provider)
    if fn is None:
        print(f"[Holiday] 未知 provider '{provider}'，回退 china_timor")
        fn = _query_timor

    # _query_timor 不接 region；_query_nager 接 country
    if provider == "nager":
        return await fn(d, region)
    return await fn(d)


async def is_workday(d: date | None = None,
                     provider: str = "china_timor",
                     region: str = "",
                     treat_makeup_as_workday: bool = True) -> bool:
    """便捷判断：今天是否应按工作日执行（节假日 API 不通时默认 True 兜底执行）。"""
    t = await get_day_type(d, provider, region)
    if t == "workday":
        return True
    if t == "makeup":
        return treat_makeup_as_workday
    if t in ("weekend", "holiday"):
        return False
    return True
