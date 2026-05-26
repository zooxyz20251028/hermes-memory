"""Engram data model — the core memory unit.

Frozen dataclass for immutability (ECC rule).
Type-annotated, validated via Pydantic.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class Engram(BaseModel):
    """A single memory engram — the core data unit.

    Frozen (immutable) by default — create new instances for changes.
    """

    model_config = {"frozen": True}

    # ── Identification ──
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID v7, auto-generated if omitted")
    statement: str = Field(..., min_length=1, description="Memory content")
    content_hash: str = Field(..., description="SHA-256 of statement for dedup")

    # ── Classification ──
    domain: str = Field(..., description="Classification domain")
    agent: str = Field(..., description="Owning agent")
    source: str = Field(..., description="Creation source")

    # ── Activation & Decay ──
    retrieval_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: int = Field(..., description="Unix ms")
    last_accessed_at: int = Field(..., description="Unix ms")
    access_count: int = Field(default=1, ge=0)
    last_feedback_at: int | None = Field(default=None, description="Unix ms")

    # ── Commitment & Priority ──
    commitment: str = Field(default="exploring", description="Injection priority")
    pinned: bool = Field(default=False)

    # ── Tags & Feedback ──
    tags: list[str] = Field(default_factory=list)
    feedback_signals: dict[str, int] = Field(default_factory=dict)
    circuit_breaker: dict[str, Any] | None = Field(default=None)

    # ── Associations ──
    associations: list[str] = Field(default_factory=list)

    # ── Embedding ──
    embedding: list[float] | None = Field(default=None)
    embedding_model: str | None = Field(default=None)

    # ── Structured Facts ──
    fact_key: str | None = Field(default=None)
    fact_value: str | None = Field(default=None)
    valid_from: int | None = Field(default=None, description="Unix ms")
    valid_to: int | None = Field(default=None, description="Unix ms, None = still valid")

    # ── Validators ──

    VALID_DOMAINS: ClassVar[frozenset[str]] = frozenset({"user", "project", "reference", "workflow", "session", "general"})
    VALID_COMMITMENTS: ClassVar[frozenset[str]] = frozenset({"locked", "decided", "leaning", "exploring"})
    VALID_SOURCES: ClassVar[frozenset[str]] = frozenset({"memory_tool", "session_learn", "feedback", "imported", "deep_learn"})
    VALID_AGENTS: ClassVar[frozenset[str]] = frozenset({"hermes", "openclaw", "orchestrator"})

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        if v not in cls.VALID_DOMAINS:
            msg = f"Invalid domain: {v!r}. Valid: {sorted(cls.VALID_DOMAINS)}"
            raise ValueError(msg)
        return v

    @field_validator("commitment")
    @classmethod
    def commitment_must_be_valid(cls, v: str) -> str:
        if v not in cls.VALID_COMMITMENTS:
            msg = f"Invalid commitment: {v!r}. Valid: {sorted(cls.VALID_COMMITMENTS)}"
            raise ValueError(msg)
        return v

    @field_validator("source")
    @classmethod
    def source_must_be_valid(cls, v: str) -> str:
        if v not in cls.VALID_SOURCES:
            msg = f"Invalid source: {v!r}. Valid: {sorted(cls.VALID_SOURCES)}"
            raise ValueError(msg)
        return v

    @field_validator("agent")
    @classmethod
    def agent_must_be_valid(cls, v: str) -> str:
        if v not in cls.VALID_AGENTS:
            msg = f"Invalid agent: {v!r}. Valid: {sorted(cls.VALID_AGENTS)}"
            raise ValueError(msg)
        return v

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Statement must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_temporal_consistency(self) -> "Engram":
        """Check valid_from <= valid_to if both set."""
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_from > self.valid_to:
                msg = "valid_from must be before valid_to"
                raise ValueError(msg)
        if self.created_at > self.last_accessed_at:
            msg = "last_accessed_at must be >= created_at"
            raise ValueError(msg)
        return self
