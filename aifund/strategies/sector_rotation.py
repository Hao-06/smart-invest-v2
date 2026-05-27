"""**行业 ETF 轮动策略** —— 用行业 ETF（而非个股）做轮动。

适用场景：
- 主线行情明确的牛市（资金集中追逐特定板块）
- 板块分化大（行业贝塔超过个股阿尔法）

设计原理（更优雅）：
- 用 **15 个主流行业 ETF** 代替「先选行业再选股」
- 计算每个 ETF 过去 20 日涨幅，选 top 3 持有
- 优点：ETF 自身分散 + 数据量小 + 不需要拉行业映射 + 一次 ADX 信号

候选行业 ETF（包含主板宽基 + 行业主题）：
- 510300 沪深300 / 510500 中证500 / 588000 科创50
- 159949 创业板50 / 512760 半导体 / 512690 主要消费
- 512170 中证医疗 / 515030 新能源 / 512000 券商
- 512800 银行 / 512200 房地产 / 515050 5G 通信
- 515700 新能车 / 512170 医疗 / 515790 光伏
"""
from __future__ import annotations

from datetime import date, timedelta

from aifund.data import sources
from aifund.strategies.base import BaseStrategy, StrategyOutput

#: 候选行业 / 主题 ETF 池
SECTOR_ETFS: dict[str, str] = {
    "510300": "沪深300",
    "510500": "中证500",
    "588000": "科创50",
    "159949": "创业板50",
    "512760": "半导体",
    "512690": "主要消费",
    "512170": "医疗",
    "515030": "新能源",
    "512000": "券商",
    "512800": "银行",
    "512200": "房地产",
    "515050": "5G通信",
    "515700": "新能车",
    "515790": "光伏",
    "515880": "通信",
}


class SectorRotationStrategy(BaseStrategy):
    """**行业 ETF 轮动** —— 选过去 20 日动量最强 3 个行业 ETF。"""

    name = "sector_rotation"
    description = "行业 ETF 轮动 —— 选过去 20 日动量最强 3 个行业 ETF（适合主线明确牛市）"
    suitable_regimes = ["trending_bull", "breakout"]

    def __init__(
        self,
        pipeline,
        top_sectors: int = 3,
        momentum_window: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(pipeline, **kwargs)
        self.top_sectors = top_sectors
        self.momentum_window = momentum_window

    def select(self, as_of: date) -> StrategyOutput:
        scores: dict[str, float] = {}
        for etf_code in SECTOR_ETFS:
            try:
                df = sources.get_price_history(
                    etf_code,
                    as_of - timedelta(days=self.momentum_window + 15),
                    as_of,
                    asset_type="etf",
                )
                if df is None or len(df) < self.momentum_window + 1:
                    continue
                df = df.sort_values("date").reset_index(drop=True)
                close = df["close"].astype(float)
                ret = float(close.iloc[-1] / close.iloc[-self.momentum_window - 1] - 1)
                scores[etf_code] = ret
            except Exception:
                continue

        if not scores:
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning="行业 ETF 轮动 · 数据全部拉取失败",
            )

        sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_etfs[: self.top_sectors]
        picks = [code for code, _ in top]
        reasoning = "行业 ETF 轮动 · 选 " + " / ".join(
            f"{SECTOR_ETFS[code]}({code}, +{ret * 100:.1f}%)" for code, ret in top
        )

        return StrategyOutput(
            symbols=picks,
            position_ratio=1.0,
            reasoning=reasoning,
            extras={
                "strategy_type": "sector_rotation",
                "top_etfs": [{"code": c, "name": SECTOR_ETFS[c], "ret_20d": r} for c, r in top],
            },
        )
