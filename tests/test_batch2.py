"""Tests for Batch 2: injector + dream_cycle."""

from __future__ import annotations

import os
import tempfile

import pytest

from hermes_memory.models import Engram
from hermes_memory.store import MemoryStore
from hermes_memory.embedding import EmbeddingClient
from hermes_memory.searcher import Searcher
from hermes_memory.compressor import Compressor


def _fake_embed():
    return EmbeddingClient(api_key="fake")


class TestInjector:
    """selectAndSpread V2 injection pipeline tests."""

    @pytest.fixture
    async def store(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test_injector.db")
        s = MemoryStore(db_path)
        await s.initialize()

        now_ms = 1_700_000 * 1000
        engrams = [
            Engram(
                statement="马兴堂偏爱使用DeepSeek V4 Flash模型",
                content_hash="h1", domain="user", agent="hermes",
                source="memory_tool", created_at=now_ms, last_accessed_at=now_ms,
                tags=["模型", "偏好"], commitment="decided",
            ),
            Engram(
                statement="宠物尿垫报价公式: (材料成本+生产费用)×1.05",
                content_hash="h2", domain="project", agent="hermes",
                source="session_learn", created_at=now_ms, last_accessed_at=now_ms,
                tags=["报价", "宠物尿垫"], commitment="locked",
                fact_key="pricing_formula",
            ),
            Engram(
                statement="NAS挂载于/Volumes/NAS/，Synology Drive同步",
                content_hash="h3", domain="reference", agent="hermes",
                source="memory_tool", created_at=now_ms, last_accessed_at=now_ms,
                tags=["NAS", "基础设施"], commitment="locked",
            ),
        ]
        for eng in engrams:
            await s.learn(eng)
        return s

    async def test_build_context_returns_string(self, store):
        from hermes_memory.injector import Injector

        searcher = Searcher(store, _fake_embed())
        compressor = Compressor()
        injector = Injector(store, searcher, compressor, _fake_embed())

        ctx = await injector.build_context("DeepSeek 模型偏好", max_tokens=500, paths=("fts5",))
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    async def test_build_context_respects_budget(self, store):
        from hermes_memory.injector import Injector

        searcher = Searcher(store, _fake_embed())
        compressor = Compressor()
        injector = Injector(store, searcher, compressor, _fake_embed())

        ctx = await injector.build_context("报价", max_tokens=50, paths=("fts5",))
        # Very tight budget — should still produce output
        assert isinstance(ctx, str)

    async def test_post_process_ranks_by_importance(self, store):
        from hermes_memory.injector import Injector

        searcher = Searcher(store, _fake_embed())
        compressor = Compressor()
        injector = Injector(store, searcher, compressor, _fake_embed())

        now_ms = 1_700_000 * 1000
        stale_ms = now_ms - 180 * 86400 * 1000  # 180 days ago
        engrams = [
            Engram(
                statement="locked rule", content_hash="hl",
                domain="user", agent="hermes", source="memory_tool",
                created_at=now_ms, last_accessed_at=now_ms,
                commitment="locked",
            ),
            Engram(
                statement="stale exploring", content_hash="hs",
                domain="general", agent="hermes", source="memory_tool",
                created_at=stale_ms, last_accessed_at=stale_ms,
                commitment="exploring",
            ),
        ]
        scored = injector._post_process(engrams)
        # locked should rank higher than stale exploring
        assert scored[0].commitment == "locked"


class TestDreamCycle:
    """7-phase nightly maintenance tests."""

    @pytest.fixture
    async def store(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test_dream.db")
        s = MemoryStore(db_path)
        await s.initialize()

        now_ms = 1_700_000 * 1000
        stale_ms = now_ms - 365 * 86400 * 1000  # 1 year ago

        engrams = [
            Engram(
                statement="fresh engram with facts",
                content_hash="h1", domain="project", agent="hermes",
                source="session_learn", created_at=now_ms, last_accessed_at=now_ms,
                tags=["报价", "公式"], commitment="decided",
                fact_key="pricing", fact_value="(材料+生产)×1.05",
            ),
            Engram(
                statement="stale low-strength engram",
                content_hash="h2", domain="general", agent="hermes",
                source="memory_tool", created_at=stale_ms, last_accessed_at=stale_ms,
                retrieval_strength=0.01, access_count=0, commitment="exploring",
            ),
            Engram(
                statement="another fresh engram with overlapping tags",
                content_hash="h3", domain="project", agent="hermes",
                source="session_learn", created_at=now_ms, last_accessed_at=now_ms,
                tags=["报价", "宠物尿垫"], commitment="decided",
            ),
        ]
        for eng in engrams:
            await s.learn(eng)
        return s

    async def test_full_cycle_returns_report(self, store):
        from hermes_memory.dream_cycle import DreamCycle

        dreamer = DreamCycle(store, _fake_embed())
        report = await dreamer.run()
        assert "summary" in report
        assert "consolidated" in report
        assert "pruned" in report
        assert "facts_extracted" in report
        assert "cross_refs" in report

    async def test_facts_extraction(self, store):
        from hermes_memory.dream_cycle import DreamCycle

        dreamer = DreamCycle(store, _fake_embed())
        report = await dreamer.run()
        # At least 1 fact should be extracted (from the engram with fact_key)
        assert report["facts_extracted"] >= 1

    async def test_cross_ref_builds_associations(self, store):
        from hermes_memory.dream_cycle import DreamCycle

        dreamer = DreamCycle(store, _fake_embed())
        report = await dreamer.run()
        # Two engrams share "报价" tag → should create at least 1 association
        assert report["cross_refs"] >= 1

    async def test_orient_reports_health(self, store):
        from hermes_memory.dream_cycle import DreamCycle

        dreamer = DreamCycle(store, _fake_embed())
        report = await dreamer.run()
        orient = report["orient"]
        assert orient["total"] == 3
        assert "health_score" in orient
