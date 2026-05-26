"""Tests for Batch 1 modules: decay, intention, token_budget."""

from __future__ import annotations

import math

import pytest

from hermes_memory.models import Engram


class TestDecay:
    """ACT-R + Weibull dual-curve decay tests."""

    def test_act_r_short_term(self):
        """ACT-R should dominate in first 10 days."""
        from hermes_memory.decay import compute_retrieval_strength

        now_ms = 1_000_000 * 1000
        created = now_ms - 5 * 86400 * 1000  # 5 days ago
        eng = Engram(
            statement="test", content_hash="h1",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, commitment="decided",
        )
        strength = compute_retrieval_strength(eng, now_ms)
        # After 5 days, ACT-R should be ~0.78, blended should be close
        assert 0.5 < strength <= 1.0

    def test_weibull_long_term(self):
        """Weibull should provide a floor beyond 30 days."""
        from hermes_memory.decay import compute_retrieval_strength

        now_ms = 1_000_000 * 1000
        created = now_ms - 60 * 86400 * 1000  # 60 days ago
        eng = Engram(
            statement="test", content_hash="h2",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, commitment="exploring",
        )
        strength = compute_retrieval_strength(eng, now_ms)
        # After 60 days, should still be > 0.01 (MIN_STRENGTH)
        assert strength >= 0.01

    def test_locked_slower_decay(self):
        """Locked memories should decay much slower than exploring."""
        from hermes_memory.decay import compute_retrieval_strength

        now_ms = 1_000_000 * 1000
        created = now_ms - 30 * 86400 * 1000  # 30 days ago

        locked = Engram(
            statement="locked", content_hash="hl",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, commitment="locked",
        )
        exploring = Engram(
            statement="exploring", content_hash="he",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, commitment="exploring",
        )
        locked_strength = compute_retrieval_strength(locked, now_ms)
        exploring_strength = compute_retrieval_strength(exploring, now_ms)
        assert locked_strength > exploring_strength

    def test_access_bonus(self):
        """High access count should partially offset decay."""
        from hermes_memory.decay import compute_retrieval_strength

        now_ms = 1_000_000 * 1000
        created = now_ms - 90 * 86400 * 1000

        frequent = Engram(
            statement="freq", content_hash="hf",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, access_count=100, commitment="decided",
        )
        rare = Engram(
            statement="rare", content_hash="hr",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=1.0, access_count=1, commitment="decided",
        )
        assert compute_retrieval_strength(frequent, now_ms) > compute_retrieval_strength(rare, now_ms)

    def test_pinned_not_stale(self):
        """Pinned engrams should never be marked stale."""
        from hermes_memory.decay import is_stale

        now_ms = 1_000_000 * 1000
        created = now_ms - 365 * 86400 * 1000  # 1 year ago
        eng = Engram(
            statement="pinned", content_hash="hp",
            domain="user", agent="hermes", source="memory_tool",
            created_at=created, last_accessed_at=now_ms,
            retrieval_strength=0.01, pinned=True,
        )
        assert not is_stale(eng, now_ms)


class TestIntention:
    """Zero-LLM query intent classification."""

    def test_preference_intent(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("马兴堂偏爱什么模型")
        assert result.intent == IntentType.USER_PREFERENCE
        assert result.bm25_weight > 1.0

    def test_temporal_intent(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("2026年5月的报价记录")
        assert result.intent == IntentType.TEMPORAL

    def test_factual_intent(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("报价公式的定义是什么")
        assert result.intent == IntentType.FACTUAL

    def test_project_intent(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("项目的部署流程")
        assert result.intent == IntentType.PROJECT

    def test_general_default(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("hello world")
        assert result.intent == IntentType.GENERAL

    def test_empty_query(self):
        from hermes_memory.intention import classify_query, IntentType
        result = classify_query("  ")
        assert result.intent == IntentType.GENERAL


class TestTokenBudget:
    """Token-budget-controlled injection."""

    def _make_engram(self, id_: str, statement: str, commitment: str = "decided", pinned: bool = False) -> Engram:
        return Engram(
            id=id_, statement=statement, content_hash=f"h{id_}",
            domain="user", agent="hermes", source="memory_tool",
            created_at=0, last_accessed_at=0,
            commitment=commitment, pinned=pinned,
        )

    def test_pinned_first(self):
        from hermes_memory.token_budget import select_for_injection

        engs = [
            self._make_engram("a", "普通条目A", "exploring"),
            self._make_engram("b", "置顶条目B", "decided", pinned=True),
            self._make_engram("c", "普通条目C", "exploring"),
        ]
        selected = select_for_injection(engs, max_tokens=500)
        assert selected[0].id == "b"  # pinned first

    def test_commitment_ordering(self):
        from hermes_memory.token_budget import select_for_injection

        engs = [
            self._make_engram("e", "探索中", "exploring"),
            self._make_engram("l", "锁定规则", "locked"),
            self._make_engram("d", "已决定", "decided"),
            self._make_engram("le", "倾向中", "leaning"),
        ]
        selected = select_for_injection(engs, max_tokens=500)
        commitments = [e.commitment for e in selected]
        assert commitments[0] == "locked"
        assert commitments[-1] == "exploring"

    def test_budget_respected(self):
        from hermes_memory.token_budget import select_for_injection, estimate_tokens

        engs = [
            self._make_engram("x", "这是一个非常非常非常非常非常非常非常非常非常长的测试语句用来验证token预算限制功能是否正常工作")
            for _ in range(20)
        ]
        selected = select_for_injection(engs, max_tokens=100)
        total = sum(estimate_tokens(e.statement) for e in selected)
        assert total <= 100

    def test_budget_stats(self):
        from hermes_memory.token_budget import budget_stats

        engs = [self._make_engram(str(i), f"条目{i}") for i in range(10)]
        stats = budget_stats(engs, max_tokens=100)
        assert stats["selected"] > 0
        assert stats["fill_pct"] > 0
