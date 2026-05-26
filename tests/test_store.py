"""Tests for hermes_memory.store — SQLite CRUD + WAL + dedup."""

from __future__ import annotations

import uuid

import pytest

from hermes_memory.models import Engram
from hermes_memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    """Create a temporary MemoryStore for testing."""
    db_path = str(tmp_path / "test_memory.db")
    s = MemoryStore(db_path=db_path)
    await s.initialize()
    yield s
    await s.close()


class TestMemoryStoreInit:
    """Store initialization and schema creation."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, store: MemoryStore):
        """After init, tables should exist."""
        conn = store._conn_or_raise
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] async for row in cursor}
        assert "engrams" in tables
        assert "associations" in tables
        assert "facts" in tables


class TestMemoryStoreLearn:
    """Engram creation — learn()."""

    @pytest.mark.asyncio
    async def test_learn_engram(self, store: MemoryStore):
        """Learn a minimal engram and verify it's stored."""
        now = 1700000000000
        e = Engram(
            statement="User prefers concise responses",
            content_hash="abc123",
            domain="user",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        assert stored.id is not None
        assert stored.statement == "User prefers concise responses"
        assert stored.commitment == "exploring"

    @pytest.mark.asyncio
    async def test_learn_dedup_by_content_hash(self, store: MemoryStore):
        """Same content_hash should update access_count, not insert new row."""
        now = 1700000000000
        e1 = Engram(
            statement="Dedup test",
            content_hash="dedup1",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        await store.learn(e1)
        e2 = Engram(
            statement="Dedup test",
            content_hash="dedup1",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e2)
        assert stored.access_count >= 2  # bumped by dedup
        # Only one row
        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_learn_multiple_engrams(self, store: MemoryStore):
        """Learn multiple engrams and verify count."""
        now = 1700000000000
        for i in range(5):
            e = Engram(
                statement=f"Memory {i}",
                content_hash=f"multi{i}",
                domain="general",
                agent="hermes",
                source="session_learn",
                created_at=now + i,
                last_accessed_at=now + i,
            )
            await store.learn(e)
        assert await store.count() == 5


class TestMemoryStoreRecall:
    """Engram retrieval — recall()."""

    @pytest.mark.asyncio
    async def test_recall_by_id(self, store: MemoryStore):
        """Recall an engram by its ID."""
        now = 1700000000000
        e = Engram(
            statement="find me",
            content_hash="findme",
            domain="project",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        recalled = await store.recall(stored.id)
        assert recalled is not None
        assert recalled.statement == "find me"

    @pytest.mark.asyncio
    async def test_recall_nonexistent(self, store: MemoryStore):
        """Recall non-existent ID should return None."""
        result = await store.recall("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_recall_bumps_access_count(self, store: MemoryStore):
        """Recalling an engram should increment access_count."""
        now = 1700000000000
        e = Engram(
            statement="bump me",
            content_hash="bump",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        await store.recall(stored.id)
        recalled = await store.recall(stored.id)
        assert recalled.access_count >= 2  # initial 1 + 2 recalls


class TestMemoryStoreFeedback:
    """Feedback signals."""

    @pytest.mark.asyncio
    async def test_positive_feedback(self, store: MemoryStore):
        """Positive feedback should increase retrieval_strength."""
        now = 1700000000000
        e = Engram(
            statement="good memory",
            content_hash="good",
            domain="general",
            agent="hermes",
            source="memory_tool",
            retrieval_strength=0.5,
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        updated = await store.feedback(stored.id, "positive")
        assert updated.retrieval_strength == pytest.approx(0.8)  # 0.5 + 0.3

    @pytest.mark.asyncio
    async def test_negative_feedback_triggers_circuit_breaker(self, store: MemoryStore):
        """3 negative feedbacks should lock the engram."""
        now = 1700000000000
        e = Engram(
            statement="bad memory",
            content_hash="bad",
            domain="general",
            agent="hermes",
            source="feedback",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        updated = stored
        for _ in range(3):
            updated = await store.feedback(stored.id, "negative")
        assert updated.circuit_breaker is not None
        assert updated.circuit_breaker["failures"] >= 3

    @pytest.mark.asyncio
    async def test_feedback_on_nonexistent_raises(self, store: MemoryStore):
        """Feedback on non-existent engram should raise KeyError."""
        with pytest.raises(KeyError):
            await store.feedback("no-such-id", "positive")


class TestMemoryStoreForget:
    """Engram deletion — forget()."""

    @pytest.mark.asyncio
    async def test_forget_engram(self, store: MemoryStore):
        """Forget an engram and verify it's gone."""
        now = 1700000000000
        e = Engram(
            statement="delete me",
            content_hash="delete",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        stored = await store.learn(e)
        await store.forget(stored.id)
        assert await store.recall(stored.id) is None
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_forget_nonexistent_no_error(self, store: MemoryStore):
        """Forgetting non-existent ID should not raise."""
        await store.forget("no-such")  # should not raise


class TestMemoryStoreConsolidate:
    """Consolidation — merging similar engrams."""

    @pytest.mark.asyncio
    async def test_consolidate_empty(self, store: MemoryStore):
        """Consolidate with no engrams should not error."""
        result = await store.consolidate()
        assert result["merged"] == 0
        assert result["pruned"] == 0
