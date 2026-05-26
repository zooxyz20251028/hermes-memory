"""Tests for hermes_memory.health — memory system health scoring."""

from __future__ import annotations

import os
import tempfile

import pytest

from hermes_memory.models import Engram
from hermes_memory.store import MemoryStore
from hermes_memory.health import HealthChecker


class TestHealth:
    """Health scoring tests."""

    @pytest.fixture
    async def store(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test_health.db")
        s = MemoryStore(db_path)
        await s.initialize()

        now_ms = 1_700_000 * 1000
        fresh_ms = now_ms  # accessed now
        stale_ms = now_ms - 60 * 86400 * 1000  # 60 days ago

        engrams = [
            Engram(
                statement="Fresh linked engram with embedding",
                content_hash="h1", domain="user", agent="hermes",
                source="memory_tool", created_at=fresh_ms,
                last_accessed_at=fresh_ms, commitment="decided",
                associations=["id2"], tags=["test"],
            ),
            Engram(
                statement="Stale engram with no embedding or links",
                content_hash="h2", domain="general", agent="hermes",
                source="memory_tool", created_at=stale_ms,
                last_accessed_at=stale_ms, commitment="exploring",
            ),
            Engram(
                statement="Fresh but broken engram",
                content_hash="h3", domain="project", agent="hermes",
                source="session_learn", created_at=fresh_ms,
                last_accessed_at=fresh_ms, commitment="leaning",
                circuit_breaker={"failures": 5, "locked_until": now_ms + 3600000},
            ),
        ]
        for eng in engrams:
            await s.learn(eng)
        return s

    async def test_compute_returns_score(self, store):
        checker = HealthChecker(store)
        report = await checker.compute()
        assert "score" in report
        assert "status" in report
        assert report["total"] == 3
        assert 0 <= report["score"] <= 100

    async def test_empty_store(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test_health_empty.db")
        s = MemoryStore(db_path)
        await s.initialize()
        checker = HealthChecker(s)
        report = await checker.compute()
        assert report["score"] == 100.0
        assert report["total"] == 0

    async def test_recommendations(self, store):
        checker = HealthChecker(store)
        report = await checker.compute()
        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) > 0
