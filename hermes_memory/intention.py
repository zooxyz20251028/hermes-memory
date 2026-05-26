"""
Zero-LLM query intent classification — ported from GBrain classifyQuery().

Classifies search queries into 5 intents using keyword patterns only
(no LLM call).  Each intent adjusts search path weights for optimal
retrieval.  This is a critical component of the injection pipeline:
before running multi-path search, classify the query to bias weights.

Intents:
  user_preference  →  BM25 +1.15×, "偏好/喜欢/讨厌/用/不要"
  temporal         →  facts table priority, "哪天/时间/之前/2026"
  factual          →  vector +1.10×, "是什么/定义/概念/解释"
  project          →  domain=project boost, "项目/代码/bug/deploy"
  general          →  default weights

ECC: Pure function, no side effects.  All patterns are immutable.

Usage:
  from hermes_memory.intention import classify_query
  intent = classify_query("马兴堂偏爱什么模型")
  # → QueryIntent.USER_PREFERENCE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class IntentType(Enum):
    """Query intent types for search weight adjustment."""

    USER_PREFERENCE = "user_preference"
    TEMPORAL = "temporal"
    FACTUAL = "factual"
    PROJECT = "project"
    GENERAL = "general"


@dataclass(frozen=True)
class QueryIntent:
    """Classified query intent with weight adjustments."""

    intent: IntentType
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    graph_weight: float = 1.0
    wiki_weight: float = 1.0

    # ── Keyword patterns (immutable class vars) ──
    _PREFERENCE_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "偏好", "喜欢", "不喜欢", "讨厌", "偏爱", "习惯",
        "prefer", "like", "hate", "want", "always",
        "从不", "从来不要", "永远",
    )
    _TEMPORAL_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "哪天", "什么时间", "时间", "之前", "之后", "最近",
        "when", "before", "after", "recently", "yesterday", "today",
        "上周", "下周", "昨天", "今天", "明天", "去年", "今年",
    )
    _FACTUAL_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "是什么", "什么是", "定义", "概念", "解释", "说明",
        "what is", "define", "definition", "explain", "describe",
        "原理", "机制", "架构", "流程",
    )
    _PROJECT_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "项目", "代码", "bug", "deploy", "部署", "测试",
        "project", "code", "fix", "commit", "pr", "pipeline",
        "报价", "管道", "hermes", "wiki",
    )
    _TEMPORAL_YEAR_RE: ClassVar[str] = r"20\d{2}"

    @classmethod
    def classify(cls, query: str) -> "QueryIntent":
        """Classify a search query into an intent type.

        Args:
            query: The raw search query string.

        Returns:
            QueryIntent with adjusted path weights.
        """
        import re

        q = query.lower().strip()
        if not q:
            return cls(intent=IntentType.GENERAL)

        # Check year patterns first (temporal signals)
        if re.search(cls._TEMPORAL_YEAR_RE, q):
            return cls(
                intent=IntentType.TEMPORAL,
                bm25_weight=1.0,
                vector_weight=0.8,
                wiki_weight=1.1,
            )

        # Score each intent
        scores: dict[IntentType, int] = {it: 0 for it in IntentType}

        for kw in cls._PREFERENCE_KEYWORDS:
            if kw in q:
                scores[IntentType.USER_PREFERENCE] += 1
        for kw in cls._TEMPORAL_KEYWORDS:
            if kw in q:
                scores[IntentType.TEMPORAL] += 1
        for kw in cls._FACTUAL_KEYWORDS:
            if kw in q:
                scores[IntentType.FACTUAL] += 1
        for kw in cls._PROJECT_KEYWORDS:
            if kw in q:
                scores[IntentType.PROJECT] += 1

        # Pick highest-scoring intent, default to GENERAL
        best = max(scores, key=lambda k: scores[k])

        if scores[best] == 0:
            return cls(intent=IntentType.GENERAL)

        # Build weight-adjusted intent
        weight_map = {
            IntentType.USER_PREFERENCE: cls(
                intent=best, bm25_weight=1.15, vector_weight=0.9,
            ),
            IntentType.TEMPORAL: cls(
                intent=best, bm25_weight=0.9, vector_weight=0.8, wiki_weight=1.1,
            ),
            IntentType.FACTUAL: cls(
                intent=best, vector_weight=1.1, wiki_weight=1.05,
            ),
            IntentType.PROJECT: cls(
                intent=best, bm25_weight=1.05, graph_weight=1.1,
            ),
            IntentType.GENERAL: cls(intent=best),
        }
        return weight_map[best]


# ── Convenience alias ──
classify_query = QueryIntent.classify
