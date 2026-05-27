"""LLM 抽象层：屏蔽不同大模型提供方的差异，Agent 只依赖统一接口。"""
from aifund.llm.client import LLMClient, LLMResponse, get_llm_client

__all__ = ["LLMClient", "LLMResponse", "get_llm_client"]
