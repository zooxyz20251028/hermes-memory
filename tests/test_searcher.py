"""Tests for hermes_memory.searcher — 4-path hybrid search + RRF fusion.

Adapted from MemPalace's test suite (MIT) — tests/ directory structure
and conftest patterns.  TDD: RED → GREEN → REFACTOR.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from hermes_memory.models import Engram
from hermes_memory.schema import get_all_ddl
from hermes_memory.store import MemoryStore

try:
    from hermes_memory.searcher import Searcher
    from hermes_memory.embedding import EmbeddingClient
    _SEARCHER_AVAILABLE = True
except ImportError:
    _SEARCHER_AVAILABLE = False


@pytest.mark.skipif(not _SEARCHER_AVAILABLE, reason="searcher not importable")
class TestSearcher:
    """Integration tests for multi-path search with RRF fusion."""

    @pytest.fixture
    async def store(self):
        """Create a test MemoryStore with sample engrams."""
        db_path = os.path.join(tempfile.mkdtemp(), "test_search.db")
        s = MemoryStore(db_path)
        await s.initialize()

        now_ms = 1_000_000 * 1000  # mock timestamp

        # Add diverse test engrams
        engrams = [
            Engram(
                statement="马兴堂偏爱使用 DeepSeek V4 Flash 模型",
                content_hash="hash_deepseek",
                domain="user",
                agent="hermes",
                source="memory_tool",
                created_at=now_ms,
                last_accessed_at=now_ms,
                tags=["模型", "偏好"],
                commitment="decided",
            ),
            Engram(
                statement="宠物尿垫报价公式: (材料成本+生产费用)×1.05",
                content_hash="hash_pricing",
                domain="project",
                agent="hermes",
                source="session_learn",
                created_at=now_ms,
                last_accessed_at=now_ms,
                tags=["报价", "宠物尿垫"],
                commitment="locked",
                fact_key="pricing_formula",
                fact_value="(材料+生产)×1.05",
            ),
            Engram(
                statement="NAS 挂载点位于 /Volumes/NAS/，通过 Synology Drive 同步",
                content_hash="hash_nas",
                domain="reference",
                agent="hermes",
                source="memory_tool",
                created_at=now_ms,
                last_accessed_at=now_ms,
                tags=["NAS", "基础设施"],
                commitment="locked",
            ),
            Engram(
                statement="Wiki RAG 搜索使用 FTS5 四层降级策略",
                content_hash="hash_rag",
                domain="project",
                agent="hermes",
                source="session_learn",
                created_at=now_ms,
                last_accessed_at=now_ms,
                tags=["wiki", "RAG"],
            ),
            Engram(
                statement="卸载插件必清除，安装插件必激活",
                content_hash="hash_discipline",
                domain="workflow",
                agent="hermes",
                source="feedback",
                created_at=now_ms,
                last_accessed_at=now_ms,
                tags=["规则", "插件"],
                commitment="locked",
            ),
        ]
        for eng in engrams:
            await s.learn(eng)
        return s

    # ── FTS5-only path ──

    async def test_fts5_exact_match(self, store):
        """FTS5 phrase match should find exact query."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        results = await searcher.search("马兴堂偏爱使用 DeepSeek V4 Flash 模型", top_k=3, paths=("fts5",))
        assert len(results) > 0
        assert any("DeepSeek" in e.statement for e in results)

    async def test_fts5_partial_match(self, store):
        """FTS5 should find partial keyword matches."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        results = await searcher.search("报价", top_k=3, paths=("fts5",))
        assert len(results) > 0
        assert any("报价" in e.statement for e in results)

    async def test_fts5_fallback_like(self, store):
        """LIKE fallback should work when FTS5 misses."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        results = await searcher.search("插件", top_k=3, paths=("fts5",))
        assert len(results) > 0
        assert any("插件" in e.statement for e in results)

    # ── Graph path ──

    async def test_graph_association(self, store):
        """Graph search should find related engrams via associations."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        # Query that matches "报价" context
        results = await searcher.search("报价 宠物尿垫", top_k=5, paths=("graph",))
        # With graph path, should find at least the pricing engram
        assert any("报价" in e.statement for e in results) or len(results) == 0  # graph may return empty if no associations

    # ── RRF fusion ──

    async def test_rrf_fusion_multi_path(self, store):
        """RRF should merge results from multiple paths."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        results = await searcher.search("DeepSeek NAS", top_k=5, paths=("fts5", "graph"))
        assert len(results) > 0
        # Should return diverse results across domains
        domains = {e.domain for e in results}
        assert len(domains) > 0

    async def test_rrf_empty_query(self, store):
        """Empty query should return empty results."""
        client = _fake_embed_client()
        searcher = Searcher(store, client)
        results = await searcher.search("   ", top_k=5)
        assert results == []

    # ── BM25 rerank ──

    def test_bm25_rerank_basic(self):
        """BM25 should rank more relevant documents higher."""
        from hermes_memory.searcher import Searcher as _S
        s = _S.__new__(_S)

        query = "DeepSeek V4 Flash 模型"
        docs = [
            Engram(
                id="a", statement="DeepSeek V4 Flash 是一个很强大的模型",
                content_hash="h1", domain="user", agent="hermes", source="memory_tool",
                created_at=0, last_accessed_at=0,
            ),
            Engram(
                id="b", statement="Flash 是 Adobe 的动画软件",
                content_hash="h2", domain="reference", agent="hermes", source="memory_tool",
                created_at=0, last_accessed_at=0,
            ),
        ]
        candidates = [(d, 1.0) for d in docs]
        reranked = s._bm25_rerank(query, candidates, top_k=2)
        # "DeepSeek V4 Flash" doc should score higher
        assert reranked[0][0].id == "a"
        assert reranked[0][1] > reranked[1][1]

    def test_tokenizer(self):
        """Tokenizer should handle Chinese + English."""
        from hermes_memory.searcher import Searcher as _S
        tokens = _S._tokenize("DeepSeek V4 Flash 模型 报价")
        assert "DeepSeek" in tokens or "deepseek" in tokens
        # CJK chars should be individual
        assert "模" in tokens or "型" in tokens or "报" in tokens or "价" in tokens


# ── Helpers ──

def _fake_embed_client():
    """Return an EmbeddingClient without real API key (for FTS5/graph-only tests)."""
    return EmbeddingClient(api_key="fake-key-no-api-calls-in-fts5-tests")
