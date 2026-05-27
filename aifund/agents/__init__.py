"""智能体模块：4 位分析师 Agent + 基金经理 Agent + 团队编排。

对外主要接口：
    AgentTeam         —— 4 位分析师 + 1 位基金经理 的完整投研团队
    Agent / AgentOpinion —— Agent 基类与统一意见模型
    FundManager / ManagerDecision —— 基金经理及其决策产物
    TechnicalAnalyst / FundFlowAnalyst / NewsAnalyst / RiskAnalyst —— 4 个分析师
"""
from aifund.agents.analysts import (
    EventPredictionAgent,
    FundFlowAnalyst,
    NewsAnalyst,
    RiskAnalyst,
    TechnicalAnalyst,
    ValuationAnalyst,
    build_analyst_team,
)
from aifund.agents.base import Agent, AgentOpinion
from aifund.agents.manager import FundManager, ManagerDecision
from aifund.agents.team import AgentTeam

__all__ = [
    "Agent",
    "AgentOpinion",
    "AgentTeam",
    "FundManager",
    "ManagerDecision",
    "TechnicalAnalyst",
    "FundFlowAnalyst",
    "NewsAnalyst",
    "RiskAnalyst",
    "ValuationAnalyst",
    "EventPredictionAgent",
    "build_analyst_team",
]
