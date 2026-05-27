"""自主选股模块（3 层漏斗）。

突破「Hao 手动指定候选池 → Agent 在池内择股」的限制，升级为「**全市场 →
自动漏斗 → Agent 决策**」的真自主选股能力。

3 层漏斗：
1. **第 1 层**（``factor_loader``）：用 Hao 前期多因子打分 CSV 取 top 20
2. **第 2 层**（``auto_selector``）：动量 30% + 资金流 30% + LSTM 涨停概率 40% 加权打分，取 top 8
3. **第 3 层**：交给现有的 6 Agent + R1 基金经理做最终决策

设计要点：
- 选股逻辑用**纯量化打分**（便宜、可控、可复现）
- 候选池**每周一**重选一次（真实基金做法，避免天天换池）
- 复用 Hao 前期多因子工作 → 设计书有完整「学生研究闭环」故事
"""
from aifund.stockpool.auto_selector import AutoSelector
from aifund.stockpool.factor_loader import FactorLoader
from aifund.stockpool.momentum_selector import MomentumSelector

__all__ = ["AutoSelector", "FactorLoader", "MomentumSelector"]
