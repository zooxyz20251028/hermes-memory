"""Tests for hermes_memory.analyzer — LLM-driven session analysis."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from hermes_memory.models import Engram
from hermes_memory.store import MemoryStore
from hermes_memory.analyzer import Analyzer, SessionContext


class TestAnalyzer:
    """Tests for regex fallback and LLM analysis integration."""

    @pytest.fixture
    async def store(self):
        """Create a test MemoryStore."""
        db_path = os.path.join(tempfile.mkdtemp(), "test_analyzer.db")
        s = MemoryStore(db_path)
        await s.initialize()
        return s

    async def test_regex_extract_correction(self, store):
        """Regex fallback should extract corrections."""
        analyzer = Analyzer(store, llm_fn=None)
        ctx = SessionContext(
            session_id="test-001",
            messages=[
                {"role": "user", "content": "用python3不是python"},
                {"role": "assistant", "content": "好的记住了"},
            ],
        )
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) > 0
        assert any("python" in e.statement.lower() for e in engrams)

    async def test_regex_extract_decision(self, store):
        """Regex should extract decisions."""
        analyzer = Analyzer(store, llm_fn=None)
        ctx = SessionContext(
            session_id="test-002",
            messages=[
                {"role": "user", "content": "我们决定使用RRF混合搜索替代纯向量"},
            ],
        )
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) > 0
        assert any("RRF" in e.statement for e in engrams)

    async def test_regex_extract_preference(self, store):
        """Regex should extract user preferences."""
        analyzer = Analyzer(store, llm_fn=None)
        ctx = SessionContext(
            session_id="test-003",
            messages=[
                {"role": "user", "content": "我偏爱使用miniMax模型"},
            ],
        )
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) > 0

    async def test_llm_extract(self, store):
        """LLM extraction should parse JSON from mock LLM."""
        mock_json = json.dumps({
            "learnings": [
                {
                    "statement": "用户要求所有报价不显示含税价格",
                    "domain": "workflow",
                    "commitment": "locked",
                    "tags": ["报价", "规则"],
                    "confidence": 0.9,
                    "type": "correction",
                },
                {
                    "statement": "DeepSeek V4 Flash 免费渠道使用 Nous Portal",
                    "domain": "reference",
                    "commitment": "decided",
                    "tags": ["模型", "API"],
                    "confidence": 0.85,
                    "type": "fact",
                },
            ]
        })

        async def mock_llm(prompt: str) -> str:
            return mock_json

        analyzer = Analyzer(store, llm_fn=mock_llm)
        ctx = SessionContext(
            session_id="test-004",
            messages=[{"role": "user", "content": "test"}],
        )
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) == 2
        assert engrams[0].commitment == "locked"
        assert engrams[1].domain == "reference"

    async def test_llm_bad_json(self, store):
        """Bad JSON from LLM should not crash."""
        async def bad_llm(prompt: str) -> str:
            return "not json at all!"

        analyzer = Analyzer(store, llm_fn=bad_llm)
        ctx = SessionContext(
            session_id="test-005",
            messages=[{"role": "user", "content": "test"}],
        )
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) == 0  # fallback handles gracefully

    async def test_confidence_filter(self, store):
        """Low-confidence candidates should be filtered out."""
        mock_json = json.dumps({
            "learnings": [
                {
                    "statement": "High confidence fact",
                    "domain": "project",
                    "commitment": "decided",
                    "tags": ["test"],
                    "confidence": 0.9,
                    "type": "fact",
                },
                {
                    "statement": "Low confidence guess",
                    "domain": "general",
                    "commitment": "exploring",
                    "tags": ["test"],
                    "confidence": 0.3,
                    "type": "fact",
                },
            ]
        })

        async def mock_llm(prompt: str) -> str:
            return mock_json

        analyzer = Analyzer(store, llm_fn=mock_llm)
        ctx = SessionContext(session_id="test-006", messages=[{"role": "user", "content": "x"}])
        engrams = await analyzer.analyze_session(ctx)
        assert len(engrams) == 1
        assert "High" in engrams[0].statement
