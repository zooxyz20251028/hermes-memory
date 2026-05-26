"""Tests for hermes_memory.compressor — AAAK-style compression."""

from __future__ import annotations

import pytest

from hermes_memory.models import Engram
from hermes_memory.compressor import Compressor


class TestCompressor:
    def setup_method(self):
        self.c = Compressor()

    def test_compress_basic(self):
        """Basic engram compression should produce compact string."""
        eng = Engram(
            statement="马兴堂偏爱使用DeepSeek V4 Flash模型",
            content_hash="h1",
            domain="user",
            agent="hermes",
            source="memory_tool",
            created_at=0,
            last_accessed_at=0,
            tags=["模型", "偏好"],
            commitment="decided",
        )
        result = self.c.compress(eng)
        assert "DeepSeek" in result
        assert "模型" in result
        assert "✅" in result

    def test_compress_locked_flag(self):
        """Locked commitment should get CORE flag."""
        eng = Engram(
            statement="NAS挂载点位于/Volumes/NAS/",
            content_hash="h2",
            domain="reference",
            agent="hermes",
            source="memory_tool",
            created_at=0,
            last_accessed_at=0,
            tags=["NAS", "基础设施"],
            commitment="locked",
        )
        result = self.c.compress(eng)
        assert "🔒" in result
        assert "CORE" in result

    def test_compress_fact_key(self):
        """Structured fact should get TECHNICAL flag."""
        eng = Engram(
            statement="宠物尿垫报价公式: (材料成本+生产费用)×1.05",
            content_hash="h3",
            domain="project",
            agent="hermes",
            source="session_learn",
            created_at=0,
            last_accessed_at=0,
            tags=["报价"],
            commitment="locked",
            fact_key="pricing_formula",
            fact_value="(材料+生产)×1.05",
        )
        result = self.c.compress(eng)
        assert "TECHNICAL" in result

    def test_compress_detects_decisions(self):
        """Decision-like keywords should trigger DECISION flag."""
        eng = Engram(
            statement="我们决定使用RRF混合搜索替代加权融合",
            content_hash="h4",
            domain="project",
            agent="hermes",
            source="session_learn",
            created_at=0,
            last_accessed_at=0,
            commitment="decided",
        )
        result = self.c.compress(eng)
        assert "DECISION" in result

    def test_compress_truncates_long(self):
        """Very long statements should be truncated."""
        eng = Engram(
            statement="这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的语句用来测试截断功能",
            content_hash="h5",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=0,
            last_accessed_at=0,
        )
        result = self.c.compress(eng)
        assert len(result) < 150
        assert "..." in result

    def test_compress_batch(self):
        """Batch compression should respect token budget."""
        engs = [
            Engram(
                statement=f"记忆条目 {i}: 这是一个非常重要的测试语句",
                content_hash=f"hb{i}",
                domain="project",
                agent="hermes",
                source="memory_tool",
                created_at=0,
                last_accessed_at=0,
                commitment="decided" if i % 2 == 0 else "exploring",
            )
            for i in range(20)
        ]
        result = self.c.compress_batch(engs, max_tokens=100)
        lines = result.split("\n")
        # Should fit within token budget
        assert len(lines) > 0
        assert len(lines) <= 15  # With 100 token budget, expect ~5-15 entries
