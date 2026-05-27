"""LLM 客户端抽象层（角色感知 + 多模型）。

设计目标：Agent 代码只声明「我需要哪种角色的模型」，不关心底层是哪个具体型号。
角色定义：
- ``"fast"``：高频、结构化输出稳定 → 默认 DeepSeek-V3（``deepseek-chat``）。
  适合 4 个分析师 Agent 并行调用。
- ``"deep"``：强推理、核心决策 → 默认 DeepSeek-R1（``deepseek-reasoner``）。
  适合基金经理 Agent 的综合权衡决策。

后续可无缝新增 Claude / MiniMax / 讯飞 等子类，只需在 ``_REGISTRY`` 中登记。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

Role = Literal["fast", "deep"]

# ---------------------------------------------------------------------------
# 响应封装
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class LLMResponse:
    """统一的 LLM 响应封装。"""

    content: str
    model: str
    role: str = "fast"
    reasoning_content: str = ""  # 推理模型的思维链（仅 R1 系列有），可用于审计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def parse_json(self) -> Any:
        """容错解析 JSON：自动剥离 markdown 代码围栏与前后噪声文本。

        Agent 输出偶尔会带解释性文字或 ```json 围栏，这里统一兜底。
        R1 推理模型不支持 JSON 模式，更依赖此函数。
        """
        text = self.content.strip()

        # 1) 优先提取代码围栏内的内容
        fenced = _JSON_FENCE.search(text)
        if fenced:
            text = fenced.group(1).strip()

        # 2) 直接尝试
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3) 截取第一个 { 或 [ 到最后一个 } 或 ] 之间的内容
        start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
        start = min(start_candidates) if start_candidates else -1
        end = max(text.rfind("}"), text.rfind("]"))
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从 LLM 响应中解析出 JSON：\n{self.content[:500]}")


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """大模型客户端抽象基类（角色感知）。"""

    provider: str = "abstract"

    def __init__(self, role: Role = "fast") -> None:
        self.role: Role = role
        self.model: str = ""  # 子类构造时填写

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """单轮对话。

        Args:
            system: 系统提示词（定义 Agent 角色与职责）。
            user: 用户提示词（本轮输入数据与任务）。
            temperature: 采样温度，None 取全局默认；推理模型会自动忽略。
            max_tokens: 最大生成长度，None 时按角色取默认值。
            json_mode: 是否请求 JSON 输出；推理模型不支持时降级为「提示要求」。
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} role={self.role} model={self.model}>"


# ---------------------------------------------------------------------------
# DeepSeek 实现（OpenAI 兼容接口，角色感知双模型）
# ---------------------------------------------------------------------------


class DeepSeekClient(LLMClient):
    """DeepSeek 大模型客户端。

    根据角色选择具体模型：
    - role="fast" → ``deepseek-chat``（DeepSeek-V3）：通用、快、JSON 模式稳定
    - role="deep" → ``deepseek-reasoner``（DeepSeek-R1）：强推理，但有若干限制
      （不支持 ``response_format`` / ``temperature`` / ``top_p`` 等参数；
      响应额外包含 ``reasoning_content`` 思维链字段）。
    本类把这些差异内化，让上层调用方式保持一致。
    """

    provider = "deepseek"

    def __init__(
        self,
        role: Role = "fast",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(role=role)
        cfg = settings.llm
        api_key = api_key or cfg.deepseek_api_key
        if not api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
            )
        self.model = model or (
            cfg.deepseek_deep_model if role == "deep" else cfg.deepseek_fast_model
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or cfg.deepseek_base_url,
            timeout=cfg.timeout,
        )

    @property
    def _is_reasoner(self) -> bool:
        """该模型是否为 R1 系列推理模型。"""
        return "reasoner" in self.model.lower() or "r1" in self.model.lower()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        cfg = settings.llm
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        if self._is_reasoner:
            # R1 推理模型：不传 temperature/response_format/top_p（官方明确不支持）。
            # JSON 模式降级为「在 system 末尾追加格式要求」，配合 parse_json 容错。
            kwargs["max_tokens"] = max_tokens or cfg.deep_max_tokens
            if json_mode:
                kwargs["messages"][0]["content"] = (
                    system + "\n\n请严格只输出合法的 JSON 对象，不要包含任何解释性文字或 markdown 围栏。"
                )
        else:
            kwargs["temperature"] = cfg.temperature if temperature is None else temperature
            kwargs["max_tokens"] = max_tokens or cfg.max_tokens
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = resp.usage
        return LLMResponse(
            content=msg.content or "",
            model=resp.model,
            role=self.role,
            reasoning_content=getattr(msg, "reasoning_content", "") or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# 工厂（按 provider + role 缓存单例）
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[LLMClient]] = {
    "deepseek": DeepSeekClient,
    # "minimax": MiniMaxClient,  # 后续可扩展
    # "claude": ClaudeClient,
}

_cached: dict[tuple[str, str], LLMClient] = {}


def get_llm_client(role: Role = "fast", provider: str | None = None) -> LLMClient:
    """按 (provider, role) 返回 LLM 客户端单例。

    Args:
        role: "fast" → 分析师层；"deep" → 基金经理层。
        provider: 显式指定提供方；None 时取 settings.llm.provider。
    """
    provider = (provider or settings.llm.provider).lower()
    if provider not in _REGISTRY:
        raise ValueError(
            f"未知的 LLM 提供方：{provider!r}，已登记：{list(_REGISTRY)}"
        )
    key = (provider, role)
    if key not in _cached:
        _cached[key] = _REGISTRY[provider](role=role)
    return _cached[key]
