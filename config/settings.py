"""全局配置。

所有可调参数集中在此，便于回测复现与赛事对接。
密钥通过 .env 注入，不写入代码库。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class LLMConfig:
    """大模型配置。LLM 层可替换，默认 DeepSeek。

    采用**角色分层**：
    - `fast` 角色：分析师 Agent 用（频繁调用，快+便宜+JSON 稳）→ DeepSeek-V3
    - `deep` 角色：基金经理 Agent 用（关键决策，强推理）→ DeepSeek-R1（推理模型）
    """

    provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "deepseek"))
    # DeepSeek（OpenAI 兼容）
    deepseek_api_key: str = field(default_factory=lambda: _get("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    # 角色对应的模型
    deepseek_fast_model: str = field(default_factory=lambda: _get("DEEPSEEK_FAST_MODEL", "deepseek-chat"))
    deepseek_deep_model: str = field(default_factory=lambda: _get("DEEPSEEK_DEEP_MODEL", "deepseek-reasoner"))
    # Anthropic（备用）
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    # 通用推理参数
    temperature: float = 0.3
    max_tokens: int = 4096
    deep_max_tokens: int = 8192  # 推理模型给更大的输出空间（含思维链）
    timeout: int = 180  # 推理模型响应慢，加长超时


@dataclass(frozen=True)
class BacktestConfig:
    """回测与交易约束。"""

    initial_capital: float = 500_000.0  # 命题规定初始虚拟资金 50 万
    commission_rate: float = 0.00025  # 佣金（双边，万 2.5）
    stamp_tax_rate: float = 0.0005  # 印花税（卖出单边，千 0.5）
    min_commission: float = 5.0  # 单笔最低佣金 5 元
    lot_size: int = 100  # A 股最小交易单元 100 股
    max_position_per_stock: float = 0.30  # 单票最大仓位占比（风控）
    max_total_position: float = 0.95  # 最大总仓位占比（留现金）


@dataclass(frozen=True)
class PathConfig:
    """路径配置。"""

    root: Path = ROOT_DIR
    data_cache: Path = ROOT_DIR / "data_cache"
    runs: Path = ROOT_DIR / "runs"  # 决策与回测产物
    docs: Path = ROOT_DIR / "docs"

    def ensure(self) -> None:
        """创建运行所需目录。"""
        for p in (self.data_cache, self.runs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# 单例
settings = Settings()
settings.paths.ensure()
