"""A股多源数据源封装。

设计要点 —— **多源容灾**：每类数据配置多个数据源（东方财富 / 腾讯财经 等），
按优先级依次尝试，主源不可用时自动回退到备用源。这样既能抵抗单一行情站点的
限流与偶发断连，也让作品在不同网络环境（评委机器）下都能稳定运行。

所有接口的参数名与返回列名均经实测核对；对外统一输出规范化的 pandas.DataFrame，
并经 `cache` 落盘为快照，保证「可审计、可复现」。
"""
from __future__ import annotations

import time
from datetime import date, datetime

import akshare as ak
import pandas as pd

from aifund.data import _net  # noqa: F401  导入即清除代理，强制直连
from aifund.data.cache import cache

# ---------------------------------------------------------------------------
# 标准列模式
# ---------------------------------------------------------------------------

#: 规范化后的日线行情列。volume 单位为「手」，amount 单位为「元」，
#: pct_change / amplitude / turnover 单位为「%」。备用源缺失的字段以 NaN 填充。
PRICE_COLUMNS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "amplitude", "pct_change", "change", "turnover",
]

_PRICE_RENAME_CN = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
    "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover",
}

# ---------------------------------------------------------------------------
# 代码与日期工具
# ---------------------------------------------------------------------------


def normalize_symbol(symbol: str) -> str:
    """规范成 6 位数字代码字符串。"""
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if not digits:
        return str(symbol)
    return digits.zfill(6)[-6:]


def market_of(symbol: str) -> str:
    """根据代码前缀判断交易所：sh / sz / bj。

    覆盖股票与场内 ETF：6/5 → 上交所，0/3/1 → 深交所，4/8 → 北交所。
    """
    s = normalize_symbol(symbol)
    head = s[0]
    if head in ("6", "5", "9"):
        return "sh"
    if head in ("0", "3", "1", "2"):
        return "sz"
    if head in ("4", "8"):
        return "bj"
    return "sh"


def _ymd(d: str | date) -> str:
    """日期统一格式化为 YYYYMMDD。"""
    if isinstance(d, str):
        return d.replace("-", "").replace("/", "")
    return d.strftime("%Y%m%d")


def _to_date(d: str | date | datetime) -> date:
    """把字符串 / datetime 统一转成 date。

    支持的字符串格式：
    - ``YYYY-MM-DD``      （ISO 标准）
    - ``YYYY/MM/DD``      （斜线分隔）
    - ``YYYYMMDD``        （8 位纯数字，AkShare 接口常用）
    """
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = str(d).strip().replace("/", "-")
    # 8 位纯数字（YYYYMMDD）→ 补成 ISO 格式
    if len(s) == 8 and s.isdigit():
        s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# 会话内负缓存
# ---------------------------------------------------------------------------
#
# 某些底层接口在当前网络环境下会持续返回空（如限流、SSL 失败、443 不通）。
# 没有负缓存的话，回测主循环每天都会重复触发同一个失败接口的全套重试 —— 整段
# 测试时间会被几秒/十几秒一次的重试拖到几分钟级。
#
# 进程内维护一个 (namespace, params_key) → 过期时间戳 的字典：失败一次后
# 在 TTL 内直接短路返回空，跨进程重启自然失效。

_NEGATIVE_TTL_SEC: float = 30 * 60  # 30 分钟内不再重撞已知失败接口
_negative_cache: dict[str, float] = {}


def _neg_key(namespace: str, params: dict) -> str:
    raw = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{namespace}::{raw}"


def _neg_hit(namespace: str, params: dict) -> bool:
    key = _neg_key(namespace, params)
    exp = _negative_cache.get(key)
    if exp is None:
        return False
    if time.time() > exp:
        _negative_cache.pop(key, None)
        return False
    return True


def _neg_set(namespace: str, params: dict, ttl: float = _NEGATIVE_TTL_SEC) -> None:
    _negative_cache[_neg_key(namespace, params)] = time.time() + ttl


# ---------------------------------------------------------------------------
# 多源重试调度
# ---------------------------------------------------------------------------


#: 单次 akshare 调用的硬超时（秒）—— 用 ThreadPoolExecutor 防 SSL read 无限阻塞
_FETCH_HARD_TIMEOUT = 30.0


def _try_sources(
    label: str,
    sources: list[tuple[str, callable]],
    retries: int = 2,
    pause: float = 1.5,
) -> tuple[pd.DataFrame, str | None]:
    """按优先级依次尝试多个数据源。

    Args:
        label: 日志标签。
        sources: [(源名称, 取数函数), ...]，按优先级排列。
        retries: 单个源的重试次数。
        pause: 重试间隔基数（秒），按次数线性递增。
    Returns:
        (DataFrame, 命中的源名称)；全部失败时返回 (空 DataFrame, None)。

    **硬超时保护**：每次 ``fn()`` 调用通过 ``ThreadPoolExecutor`` 包裹，超过
    ``_FETCH_HARD_TIMEOUT`` 秒未返回则抛 TimeoutError，触发 retry / 切换源。
    这是必要的兜底，因为某些 SSL read 可能完全不响应 socket-level timeout。
    """
    import threading

    last_err: Exception | None = None
    for idx, (name, fn) in enumerate(sources):
        for attempt in range(retries):
            # 用 daemon Thread + Event 真正硬超时（不等待子线程退出）
            result_holder: dict = {}
            done_event = threading.Event()

            def _worker():
                try:
                    result_holder["df"] = fn()
                except Exception as exc:  # noqa: BLE001
                    result_holder["err"] = exc
                finally:
                    done_event.set()

            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            finished = done_event.wait(timeout=_FETCH_HARD_TIMEOUT)
            # 注意：不等待 th 自然退出 —— daemon 线程会随主进程结束
            if not finished:
                last_err = TimeoutError(f"hard timeout {_FETCH_HARD_TIMEOUT}s")
                if attempt + 1 < retries:
                    time.sleep(pause * (attempt + 1))
                continue
            if "err" in result_holder:
                last_err = result_holder["err"]
                if attempt + 1 < retries:
                    time.sleep(pause * (attempt + 1))
                continue
            df = result_holder.get("df")
            if df is not None and not df.empty:
                if idx > 0:
                    print(f"[data] {label}: 主源不可用，已回退到「{name}」")
                return df, name
    detail = f"（{type(last_err).__name__}: {str(last_err)[:80]}）" if last_err else ""
    print(f"[data] {label}: 所有数据源均失败{detail}")
    return pd.DataFrame(), None


# ---------------------------------------------------------------------------
# 行情：日线
# ---------------------------------------------------------------------------


def _normalize_price(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """把不同来源的日线行情规范成 PRICE_COLUMNS 模式。

    各源差异：
    - 东方财富：中文列名，含成交量(手)/成交额(元)/换手率等完整字段。
    - 腾讯财经：列名 date/open/close/high/low + 第 6 列 "amount" 实为
      **成交量（单位：手）**，无成交额、换手率等字段（经实测核对）。
    """
    df = df.copy()
    if source in ("eastmoney", "etf_em"):
        df = df.rename(columns=_PRICE_RENAME_CN)
    elif source == "tencent":
        # 腾讯源的 "amount" 列实为成交量（手），并非成交额
        df = df.rename(columns={"amount": "volume"})
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in PRICE_COLUMNS:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[PRICE_COLUMNS].dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def get_price_history(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    asset_type: str = "stock",
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取 [start_date, end_date] 区间内的日线行情（前复权），列见 PRICE_COLUMNS。

    **缓存策略（关键）**：按 ``(symbol, asset_type)`` 单键缓存「2010-01-01 至今」
    的长历史；区间查询在内存中按 mask 截取。这样不同 ``as_of`` 与不同 lookback
    窗口共享同一份缓存 —— 60 个交易日的回测从「60 次拉取」降为「1 次拉取」。

    数据源优先级：股票 = 东方财富 → 腾讯财经；ETF = 东方财富。
    """
    symbol = normalize_symbol(symbol)
    start, end = _to_date(start_date), _to_date(end_date)
    cache_params = {"symbol": symbol, "asset": asset_type, "scope": "long"}

    df: pd.DataFrame | None = None
    if use_cache:
        df = cache.get("price_history", cache_params)
        if df is not None and not df.empty and "date" in df.columns:
            first = df["date"].iloc[0]
            if not isinstance(first, date):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    cached_end = df["date"].max() if (df is not None and not df.empty) else None
    needs_fetch = df is None or df.empty or cached_end is None or cached_end < end

    if needs_fetch:
        long_start = date(2010, 1, 1)
        long_end = max(end, date.today())
        s, e = _ymd(long_start), _ymd(long_end)

        if asset_type == "etf":
            mkt = market_of(symbol)
            sources_list: list[tuple[str, callable]] = [
                ("etf_em", lambda: ak.fund_etf_hist_em(
                    symbol=symbol, period="daily",
                    start_date=s, end_date=e, adjust="qfq")),
                # 腾讯财经备用：ETF 也支持，列含义与个股一致（amount 实为成交量手）
                ("tencent", lambda: ak.stock_zh_a_hist_tx(
                    symbol=f"{mkt}{symbol}",
                    start_date=s, end_date=e, adjust="qfq")),
            ]
        else:
            mkt = market_of(symbol)
            sources_list = [
                ("eastmoney", lambda: ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=s, end_date=e, adjust="qfq")),
                ("tencent", lambda: ak.stock_zh_a_hist_tx(
                    symbol=f"{mkt}{symbol}",
                    start_date=s, end_date=e, adjust="qfq")),
            ]

        raw, src = _try_sources(f"行情({symbol})", sources_list)
        if not raw.empty:
            fresh = _normalize_price(raw, src or "")
            fresh.attrs["source"] = src
            df = fresh
            if use_cache:
                cache.put("price_history", cache_params, df)
        elif df is None or df.empty:
            # 拉取失败且无旧缓存
            return pd.DataFrame(columns=PRICE_COLUMNS)
        # 否则沿用旧缓存

    # 在内存中截取请求区间
    assert df is not None
    mask = (df["date"] >= start) & (df["date"] <= end)
    result = df.loc[mask].reset_index(drop=True)
    if df.attrs.get("source"):
        result.attrs["source"] = df.attrs["source"]
    return result


# ---------------------------------------------------------------------------
# 资金面：个股资金流
# ---------------------------------------------------------------------------

_FUND_FLOW_RENAME = {
    "日期": "date", "收盘价": "close", "涨跌幅": "pct_change",
    "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct",
    "超大单净流入-净额": "xlarge_net", "大单净流入-净额": "large_net",
    "中单净流入-净额": "medium_net", "小单净流入-净额": "small_net",
}


def get_fund_flow(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """获取个股资金流历史（主力 / 超大单 / 大单 / 中单 / 小单 净流入）。"""
    symbol = normalize_symbol(symbol)
    params = {"symbol": symbol}

    if use_cache:
        cached = cache.get("fund_flow", params, ttl=12 * 3600)
        if cached is not None:
            return cached
        if _neg_hit("fund_flow", params):
            return pd.DataFrame()

    mkt = market_of(symbol)
    sources = [
        ("eastmoney", lambda: ak.stock_individual_fund_flow(stock=symbol, market=mkt)),
    ]
    raw, src = _try_sources(f"资金流({symbol})", sources)
    if raw.empty:
        _neg_set("fund_flow", params)
        return pd.DataFrame()

    df = raw.rename(columns=_FUND_FLOW_RENAME)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    keep = ["date", "close", "pct_change", "main_net", "main_net_pct",
            "xlarge_net", "large_net", "medium_net", "small_net"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["source"] = src
    if use_cache:
        cache.put("fund_flow", params, df)
    return df


# ---------------------------------------------------------------------------
# 资金面：北向资金
# ---------------------------------------------------------------------------


def get_hsgt_flow(use_cache: bool = True) -> pd.DataFrame:
    """获取北向资金历史净流入。列名因 AkShare 版本而异，做容错映射。"""
    params = {"type": "北向资金"}
    if use_cache:
        cached = cache.get("hsgt_flow", params, ttl=12 * 3600)
        if cached is not None:
            return cached

    sources = [("eastmoney", lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))]
    raw, src = _try_sources("北向资金", sources)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    rename: dict[str, str] = {}
    for col in df.columns:
        if col == "日期":
            rename[col] = "date"
        elif "当日成交净买额" in col:
            rename[col] = "net_buy"
        elif "当日资金流入" in col:
            rename[col] = "net_inflow"
        elif "历史累计净买额" in col:
            rename[col] = "cumulative_net"
        elif "持股市值" in col:
            rename[col] = "holding_value"
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.sort_values("date").reset_index(drop=True)
    df.attrs["source"] = src
    if use_cache:
        cache.put("hsgt_flow", params, df)
    return df


# ---------------------------------------------------------------------------
# 消息面：个股新闻
# ---------------------------------------------------------------------------

_NEWS_RENAME = {
    "关键词": "keyword", "新闻标题": "title", "新闻内容": "content",
    "发布时间": "publish_time", "文章来源": "source", "新闻链接": "url",
}


def get_news(symbol: str, limit: int = 20, use_cache: bool = True) -> pd.DataFrame:
    """获取个股近期新闻，按发布时间倒序，最多 limit 条。"""
    symbol = normalize_symbol(symbol)
    params = {"symbol": symbol}

    if use_cache:
        cached = cache.get("news", params, ttl=6 * 3600)  # 新闻时效性强，TTL 6 小时
        if cached is not None:
            return cached.head(limit).reset_index(drop=True)
        if _neg_hit("news", params):
            return pd.DataFrame()

    sources = [("eastmoney", lambda: ak.stock_news_em(symbol=symbol))]
    raw, src = _try_sources(f"新闻({symbol})", sources)
    if raw.empty:
        _neg_set("news", params)
        return pd.DataFrame()

    df = raw.rename(columns=_NEWS_RENAME)
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df = df.sort_values("publish_time", ascending=False)
    df = df.reset_index(drop=True)
    df.attrs["source"] = src
    if use_cache:
        cache.put("news", params, df)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 基础信息：代码名称表
# ---------------------------------------------------------------------------


def get_stock_list(use_cache: bool = True) -> pd.DataFrame:
    """获取全 A 股代码-名称表，列 = symbol, name。"""
    params = {"v": 1}
    if use_cache:
        cached = cache.get("stock_list", params, ttl=24 * 3600)
        if cached is not None:
            return cached

    def _from_spot() -> pd.DataFrame:
        spot = ak.stock_zh_a_spot_em()
        return spot[["代码", "名称"]].rename(columns={"代码": "symbol", "名称": "name"})

    def _from_code_name() -> pd.DataFrame:
        return ak.stock_info_a_code_name().rename(columns={"code": "symbol"})

    df, _ = _try_sources("代码名称表", [("spot_em", _from_spot), ("code_name", _from_code_name)])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "name"])

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[["symbol", "name"]].drop_duplicates("symbol").reset_index(drop=True)
    if use_cache:
        cache.put("stock_list", params, df)
    return df


def get_stock_info(symbol: str, use_cache: bool = True) -> dict[str, object]:
    """获取个股基本信息：股票简称、行业、总市值、流通市值、上市时间等。

    底层为东方财富个股信息接口（item/value 长表），轻量稳定，
    既用于名称兜底，也为风控 Agent 提供流通市值、行业等基本面字段。
    """
    symbol = normalize_symbol(symbol)
    params = {"symbol": symbol}

    df: pd.DataFrame | None = None
    if use_cache:
        df = cache.get("stock_info", params, ttl=7 * 24 * 3600)
        if df is None and _neg_hit("stock_info", params):
            return {}
    if df is None:
        # 该接口偶发空响应（疑似限流），适度增加重试次数与退避间隔
        raw, _ = _try_sources(
            f"个股信息({symbol})",
            [("eastmoney", lambda: ak.stock_individual_info_em(symbol=symbol))],
            retries=3,
            pause=2.5,
        )
        if raw.empty or not {"item", "value"}.issubset(raw.columns):
            _neg_set("stock_info", params)
            return {}
        df = raw[["item", "value"]].astype({"item": str, "value": str}).copy()
        if use_cache:
            cache.put("stock_info", params, df)
    return dict(zip(df["item"], df["value"]))


_VALUATION_RENAME = {
    "数据日期": "trade_date",
    "当日收盘价": "close",
    "当日涨跌幅": "pct_change",
    "总市值": "total_mv",
    "流通市值": "float_mv",
    "总股本": "total_shares",
    "流通股本": "float_shares",
    "PE(TTM)": "pe_ttm",
    "PE(静)": "pe_static",
    "市净率": "pb",
    "PEG值": "peg",
    "市现率": "pcf",
    "市销率": "ps_ttm",
}


def get_valuation_metrics(symbol: str, use_cache: bool = True) -> dict[str, object]:
    """获取估值面因子（PE/PB/PS/PEG/市值）+ 在近 1 年历史区间的分位。

    数据源：AkShare ``stock_value_em``（东方财富个股价值评估，含 PE_TTM/PE_静/PB/PEG/PCF/PS_TTM
    + 总市值/流通市值 + 总股本/流通股本的完整历史；A 股全覆盖）。

    返回字典：
    - ``latest``：最新一日的 PE_TTM、PE_静、PB、PEG、PS_TTM、PCF、总市值、流通市值
    - ``percentile_in_1y``：当前 PE_TTM/PB/PS_TTM 在近 252 个交易日的历史分位（0=最低 / 1=最高）

    失败时返回空字典，调用方按缺失处理（ValuationAnalyst 会输出 confidence=0 的「无数据」意见）。
    """
    symbol = normalize_symbol(symbol)
    params = {"symbol": symbol}

    df: pd.DataFrame | None = None
    if use_cache:
        df = cache.get("valuation", params, ttl=24 * 3600)
        if df is None and _neg_hit("valuation", params):
            return {}
    if df is None:
        raw, _ = _try_sources(
            f"估值因子({symbol})",
            [("em", lambda: ak.stock_value_em(symbol=symbol))],
            retries=2,
            pause=1.5,
        )
        if raw is None or raw.empty:
            _neg_set("valuation", params)
            return {}
        df = raw.rename(columns=_VALUATION_RENAME).copy()
        # 数值列转 float、日期列转 date
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        for c in ("close", "pct_change", "total_mv", "float_mv", "total_shares",
                 "float_shares", "pe_ttm", "pe_static", "pb", "peg", "pcf", "ps_ttm"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        if use_cache:
            cache.put("valuation", params, df)

    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    recent = df.tail(252)  # 近一年

    def _pct_rank(series: pd.Series, value: float) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty or pd.isna(value):
            return None
        return round(float((clean <= value).mean()), 3)

    def _safe(name: str) -> float | None:
        v = last.get(name)
        try:
            f = float(v)
            return round(f, 4) if not pd.isna(f) else None
        except (TypeError, ValueError):
            return None

    pe_ttm = _safe("pe_ttm")
    pe_static = _safe("pe_static")
    pb = _safe("pb")
    ps_ttm = _safe("ps_ttm")
    peg = _safe("peg")
    pcf = _safe("pcf")
    total_mv = _safe("total_mv")
    float_mv = _safe("float_mv")

    return {
        "as_of": str(last.get("trade_date") or ""),
        "latest": {
            "pe_ttm": pe_ttm,
            "pe_static": pe_static,
            "pb": pb,
            "ps_ttm": ps_ttm,
            "peg": peg,
            "pcf": pcf,
            "total_mv_wan": round(total_mv / 10000.0, 2) if total_mv else None,
            "float_mv_wan": round(float_mv / 10000.0, 2) if float_mv else None,
        },
        "percentile_in_1y": {
            "pe_ttm": _pct_rank(recent.get("pe_ttm", pd.Series(dtype=float)), pe_ttm) if pe_ttm else None,
            "pb": _pct_rank(recent.get("pb", pd.Series(dtype=float)), pb) if pb else None,
            "ps_ttm": _pct_rank(recent.get("ps_ttm", pd.Series(dtype=float)), ps_ttm) if ps_ttm else None,
        },
    }


def get_stock_name(symbol: str) -> str:
    """按代码查证券简称，查不到时回退为代码本身。

    优先查全市场代码表；失败时退化到轻量的个股信息接口。
    """
    symbol = normalize_symbol(symbol)
    lst = get_stock_list()
    if not lst.empty:
        hit = lst[lst["symbol"] == symbol]
        if not hit.empty:
            return str(hit.iloc[0]["name"])
    info = get_stock_info(symbol)
    name = info.get("股票简称")
    return str(name) if name else symbol


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------


def get_index_history(
    symbol: str = "000300",
    start_date: str | date = "2010-01-01",
    end_date: str | date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取大盘指数日线历史，默认沪深300（``000300``）。

    用于回测时作为业内标准 benchmark 对比。返回列与 ``PRICE_COLUMNS`` 一致。

    数据源优先级：东方财富 → 腾讯财经。
    """
    symbol = "".join(ch for ch in str(symbol) if ch.isdigit())
    start = _to_date(start_date)
    end = _to_date(end_date) if end_date else date.today()
    cache_params = {"symbol": symbol, "scope": "index_long"}

    df: pd.DataFrame | None = None
    if use_cache:
        df = cache.get("index_history", cache_params)
        if df is not None and not df.empty and "date" in df.columns:
            first = df["date"].iloc[0]
            if not isinstance(first, date):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    cached_end = df["date"].max() if (df is not None and not df.empty) else None
    if df is None or df.empty or cached_end is None or cached_end < end:
        long_start, long_end = date(2010, 1, 1), max(end, date.today())
        s, e = _ymd(long_start), _ymd(long_end)
        # sh / sz 前缀（沪深 300 用 sh）
        prefix = "sh" if symbol.startswith(("0", "3")) else "sh"  # 默认 sh

        def _from_em() -> pd.DataFrame:
            return ak.index_zh_a_hist(
                symbol=symbol, period="daily", start_date=s, end_date=e
            )

        def _from_em_old() -> pd.DataFrame:
            return ak.stock_zh_index_daily_em(symbol=f"{prefix}{symbol}")

        raw, src = _try_sources(
            f"指数行情({symbol})",
            [("em_hist", _from_em), ("em_daily", _from_em_old)],
        )
        if not raw.empty:
            fresh = _normalize_price(raw, "eastmoney" if src == "em_hist" else "eastmoney")
            fresh.attrs["source"] = src
            df = fresh
            if use_cache:
                cache.put("index_history", cache_params, df)
        elif df is None or df.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)

    assert df is not None
    mask = (df["date"] >= start) & (df["date"] <= end)
    result = df.loc[mask].reset_index(drop=True)
    if df.attrs.get("source"):
        result.attrs["source"] = df.attrs["source"]
    return result


def get_trade_calendar(use_cache: bool = True) -> pd.DataFrame:
    """获取 A股交易日历，列 = date。

    主源 = 新浪；退化方案 = 用基准股票（浦发银行 600000）的行情日期推导。

    **强缓存兜底**：即使 TTL 过期 + 网络拉取失败，也用旧缓存（**避免单点失败导致整个回测崩**）。
    """
    params = {"v": 1}
    cached_fresh = None  # TTL 内的缓存
    cached_stale = None  # TTL 过期但存在的缓存（兜底用）
    if use_cache:
        cached_fresh = cache.get("trade_calendar", params, ttl=24 * 3600)
        if cached_fresh is not None:
            return cached_fresh
        # 取过期缓存做兜底
        cached_stale = cache.get("trade_calendar", params, ttl=999 * 24 * 3600)

    def _from_sina() -> pd.DataFrame:
        df = ak.tool_trade_date_hist_sina().rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df

    def _from_reference() -> pd.DataFrame:
        ref = get_price_history("600000", "20100101", date.today(), use_cache=True)
        return pd.DataFrame({"date": ref["date"]})

    df, _ = _try_sources(
        "交易日历", [("sina", _from_sina), ("reference", _from_reference)]
    )
    if df.empty:
        # 拉失败 → 用过期缓存兜底（**新增**）
        if cached_stale is not None and not cached_stale.empty:
            print("[data] 交易日历：拉取失败，用过期缓存兜底")
            return cached_stale
        return pd.DataFrame(columns=["date"])

    df = df[["date"]].dropna().drop_duplicates().sort_values("date").reset_index(drop=True)
    if use_cache:
        cache.put("trade_calendar", params, df)
    return df
