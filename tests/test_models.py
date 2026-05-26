"""Tests for hermes_memory.models — Engram data model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hermes_memory.models import Engram


class TestEngramModel:
    """Engram frozen dataclass — immutability + validation."""

    def test_create_engram_minimal(self):
        """Can create an Engram with only required fields."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e = Engram(
            statement="User prefers concise responses",
            content_hash="abc123",
            domain="user",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        assert e.statement == "User prefers concise responses"
        assert e.content_hash == "abc123"
        assert e.domain == "user"
        assert e.agent == "hermes"
        assert e.source == "memory_tool"
        assert e.commitment == "exploring"  # default
        assert e.pinned is False  # default
        assert e.access_count == 1  # default
        assert e.retrieval_strength == 1.0  # default
        assert e.tags == []  # default
        assert e.feedback_signals == {}  # default
        assert e.circuit_breaker is None  # default

    def test_engram_is_frozen(self):
        """Engram should be immutable after creation."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e = Engram(
            statement="immutable test",
            content_hash="fix",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        with pytest.raises(ValidationError):
            # frozen=True — should raise on direct attribute set
            e.statement = "trying to change"

    def test_engram_default_commitment_exploring(self):
        """Default commitment should be 'exploring'."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e = Engram(
            statement="test",
            content_hash="def",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        assert e.commitment == "exploring"

    def test_engram_all_fields(self):
        """Can create an Engram with all fields populated."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e = Engram(
            id="0196c5e0-7b00-8000-8000-000000000001",
            statement="Project uses Python 3.11",
            content_hash="ghi789",
            domain="project",
            agent="openclaw",
            source="session_learn",
            retrieval_strength=0.8,
            created_at=now,
            last_accessed_at=now,
            access_count=5,
            last_feedback_at=now,
            commitment="decided",
            pinned=True,
            tags=["python", "project-config"],
            feedback_signals={"positive": 2, "negative": 0},
            circuit_breaker={"failures": 0, "locked_until": 0},
            embedding=[0.1, 0.2, 0.3],
            embedding_model="text-embedding-v4",
            associations=["id-1", "id-2"],
            fact_key="python_version",
            fact_value="3.11",
            valid_from=now,
        )
        assert e.id == "0196c5e0-7b00-8000-8000-000000000001"
        assert e.fact_key == "python_version"
        assert e.pinned is True
        assert e.embedding == [0.1, 0.2, 0.3]

    def test_engram_domain_validation(self):
        """Domain must be one of valid values."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with pytest.raises(ValueError):
            Engram(  # type: ignore[call-arg]
                statement="bad",
                content_hash="bad",
                domain="invalid_domain",
                agent="hermes",
                source="memory_tool",
                created_at=now,
                last_accessed_at=now,
            )

    def test_engram_commitment_validation(self):
        """Commitment must be one of valid values."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with pytest.raises(ValueError):
            Engram(  # type: ignore[call-arg]
                statement="bad",
                content_hash="bad",
                domain="general",
                agent="hermes",
                source="memory_tool",
                created_at=now,
                last_accessed_at=now,
                commitment="invalid_level",
            )

    def test_engram_source_validation(self):
        """Source must be one of valid values."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with pytest.raises(ValueError):
            Engram(  # type: ignore[call-arg]
                statement="bad",
                content_hash="bad",
                domain="general",
                agent="hermes",
                source="unknown_source",
                created_at=now,
                last_accessed_at=now,
            )

    def test_engram_retrieval_strength_bounds(self):
        """retrieval_strength must be 0-1."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with pytest.raises(ValueError):
            Engram(  # type: ignore[call-arg]
                statement="bad",
                content_hash="bad",
                domain="general",
                agent="hermes",
                source="memory_tool",
                created_at=now,
                last_accessed_at=now,
                retrieval_strength=1.5,
            )

    def test_engram_statement_not_empty(self):
        """Statement must not be empty."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with pytest.raises(ValueError):
            Engram(  # type: ignore[call-arg]
                statement="",
                content_hash="bad",
                domain="general",
                agent="hermes",
                source="memory_tool",
                created_at=now,
                last_accessed_at=now,
            )

    def test_engram_equality_by_id(self):
        """Two Engrams with same id should be equal."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e1 = Engram(
            id="same-id",
            statement="test",
            content_hash="a",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        e2 = Engram(
            id="same-id",
            statement="test",
            content_hash="a",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        assert e1 == e2

    def test_engram_str_representation(self):
        """String representation should be readable."""
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        e = Engram(
            statement="Hello world",
            content_hash="str",
            domain="general",
            agent="hermes",
            source="memory_tool",
            created_at=now,
            last_accessed_at=now,
        )
        s = str(e)
        assert "Hello world" in s
        assert "general" in s
