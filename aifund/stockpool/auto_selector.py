"""3 层漏斗自主选股器。

第 1 层：用 ``FactorLoader`` 取 top-20（按 Hao 多因子综合 score）
第 2 层：动量 30% + 资金流 30% + LSTM 涨停概率 40% 加权打分，取 top-8
第 3 层：返回的标的池交给现有 6 Agent + R1 基金经理做最终决策

设计要点：
- 第 1 层用 Hao 现成的 score（覆盖 2024-2025 沪深300 成分股的 293 只）
- 第 2 层每个分量都**归一化**到 [0, 1] 再加权，避免量纲差异主导
- LSTM 涨停概率给最高权重（40%）—— 因为它是 Hao 的核心训练成果
- **周频重选**：避免天天换池导致换手率爆炸
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aifund.data import sources
from aifund.data.calendar import TradeCalendar
from aifund.data.pipeline import DataPipeline
from aifund.ml import LimitUpPredictor
from aifund.stockpool.factor_loader import FactorLoader


class AutoSelector:
    """3 层漏斗自主选股 Agent。

    Args:
        pipeline: 数据管道（共享缓存与日历）。
        factor_csv_path: Hao 多因子 CSV 路径，None 时用默认路径。
        top1: 第 1 层粗筛保留数。
        top2: 第 2 层精筛保留数（即最终候选池大小）。
        weights: 第 2 层加权分数权重（动量, 资金流, LSTM）；和应为 1.0。
    """

    def __init__(
        self,
        pipeline: DataPipeline,
        factor_csv_path: Path | str | None = None,
        top1: int = 20,
        top2: int = 8,
        weights: tuple[float, float, float] = (0.30, 0.30, 0.40),
    ) -> None:
        self.pipeline = pipeline
        self.loader = FactorLoader(factor_csv_path)
        self.top1 = top1
        self.top2 = top2
        w_sum = sum(weights)
        if abs(w_sum - 1.0) > 1e-6:
            weights = tuple(w / w_sum for w in weights)  # 归一化
        self.w_momentum, self.w_fundflow, self.w_lstm = weights
        self.predictor = LimitUpPredictor()

    # ------------------------------------------------------------------
    # 第 1 层：多因子 score 粗筛
    # ------------------------------------------------------------------
    def coarse_filter(self, as_of: date) -> list[str]:
        """第 1 层：取 Hao 多因子 score top-N。"""
        top = self.loader.top_n_by_score(as_of, self.top1)
        return top["symbol"].tolist() if not top.empty else []

    # ------------------------------------------------------------------
    # 第 2 层：4 个 CSV 因子复合打分（**纯量化、不调 LSTM、不拉网络**）
    # ------------------------------------------------------------------
    def fine_filter(self, as_of: date, candidates: list[str]) -> list[dict]:
        """第 2 层：对粗筛后的标的用 4 个 CSV 因子加权综合打分。

        **关键设计**：本层 **100% 基于 CSV 历史因子**，不调用 LSTM、不拉网络数据 ——
        - 选股阶段秒级完成，不会 hang
        - LSTM 与外部数据留给「多 Agent 决策」阶段（EventPredictionAgent 仍会跑 LSTM）

        4 个因子加权打分：
        - **动量分量**（``w_momentum``，默认 30%）= ``Factor_MOM_60`` 中期动量
        - **活跃度分量**（``w_fundflow``，默认 30%）= ``Factor_Turn_20`` 近 20 日换手率
        - **估值分量**（``w_lstm`` 的一半，默认 20%）= ``Factor_BP`` 账面市值比（越高越便宜）
        - **反转惩罚**（``w_lstm`` 的一半，默认 20%）= **负** ``Factor_REV_5``（短期涨太多则惩罚）

        Returns:
            含 ``symbol``、``factor_mom``、``factor_turn``、``factor_bp``、
            ``factor_rev``、``combo_score`` 的字典列表，按 combo_score 降序。
        """
        # 一次性拿到当日所有因子（避免重复查询）
        factor_df = self.loader.factors_by_date(as_of)
        if factor_df.empty:
            return []
        factor_df = factor_df.set_index("symbol")

        rows: list[dict] = []
        for symbol in candidates:
            if symbol not in factor_df.index:
                continue
            f = factor_df.loc[symbol]

            rows.append({
                "symbol": symbol,
                "factor_mom": float(f.get("Factor_MOM_60", 0.0) or 0.0),
                "factor_turn": float(f.get("Factor_Turn_20", 0.0) or 0.0),
                "factor_bp": float(f.get("Factor_BP", 0.0) or 0.0),
                "factor_rev": float(f.get("Factor_REV_5", 0.0) or 0.0),
            })

        if not rows:
            return []

        df = pd.DataFrame(rows)

        # 各因子归一化到 [0, 1]
        def _norm(col: str) -> pd.Series:
            mn, mx = float(df[col].min()), float(df[col].max())
            return (df[col] - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(df), index=df.index)

        df["mom_n"] = _norm("factor_mom")
        df["turn_n"] = _norm("factor_turn")
        df["bp_n"] = _norm("factor_bp")
        df["rev_n"] = _norm("factor_rev")  # 反向惩罚：高的减分

        # 拆分 w_lstm 为估值 + 反转惩罚（各占 w_lstm 的一半）
        w_val = self.w_lstm / 2
        w_rev_pen = self.w_lstm / 2

        df["combo_score"] = (
            df["mom_n"] * self.w_momentum
            + df["turn_n"] * self.w_fundflow
            + df["bp_n"] * w_val
            - df["rev_n"] * w_rev_pen
        )
        df = df.sort_values("combo_score", ascending=False)
        return df.head(self.top2).to_dict("records")

    # ------------------------------------------------------------------
    # 一站式：3 层漏斗
    # ------------------------------------------------------------------
    def select(self, as_of: date, verbose: bool = False) -> list[str]:
        """对 as_of 日运行完整的 3 层漏斗，返回最终候选池标的列表。"""
        coarse = self.coarse_filter(as_of)
        if not coarse:
            if verbose:
                print(f"[selector] {as_of} 第 1 层无数据")
            return []
        fine = self.fine_filter(as_of, coarse)
        symbols = [r["symbol"] for r in fine]
        if verbose:
            print(f"[selector] {as_of} 漏斗结果（{len(coarse)} → {len(symbols)}）:")
            for r in fine:
                print(f"  {r['symbol']}  combo={r['combo_score']:.3f}  "
                      f"MOM60={r['factor_mom']:+.3f}  "
                      f"TURN20={r['factor_turn']:.3f}  "
                      f"BP={r['factor_bp']:+.3f}  "
                      f"REV5={r['factor_rev']:+.3f}")
        return symbols

    # ------------------------------------------------------------------
    # 周频预计算（供回测使用）
    # ------------------------------------------------------------------
    def precompute_weekly_pools(
        self,
        start_date: date,
        end_date: date,
        calendar: TradeCalendar | None = None,
        verbose: bool = True,
    ) -> dict[date, list[str]]:
        """预计算 [start, end] 区间内每周首个交易日的候选池。

        Returns:
            dict[周首日, [候选标的列表]]
        """
        cal = calendar or self.pipeline.calendar
        all_dates = cal.range(start_date, end_date)
        if not all_dates:
            return {}

        # 标识每个交易日所在 ISO 周（年, 周号），同周内只在首个交易日选股
        pools: dict[date, list[str]] = {}
        last_week_key: tuple[int, int] | None = None
        for d in all_dates:
            year, week, _ = d.isocalendar()
            week_key = (year, week)
            if week_key != last_week_key:
                if verbose:
                    print(f"[selector] 预计算 {d} 候选池…", flush=True)
                pools[d] = self.select(d, verbose=False)
                if verbose and pools[d]:
                    print(f"  → {pools[d]}", flush=True)
                last_week_key = week_key
        return pools
