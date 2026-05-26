"""MemoryStore — SQLite CRUD + WAL + dedup + feedback + circuit breaker.

Core data access layer for hermes-memory.
All mutations go through aiosqlite with WAL mode.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import aiosqlite

from hermes_memory.models import Engram
from hermes_memory.schema import get_all_ddl


class MemoryStore:
    """SQLite-backed memory store with WAL mode.

    Uses a single persistent connection. No nested ``async with`` on the
    connection itself — just ``await conn.execute(...)`` and explicit
    ``await conn.commit()``.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection, enable WAL, create tables."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

        for ddl in get_all_ddl():
            await self._conn.execute(ddl)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _conn_or_raise(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._conn

    async def count(self) -> int:
        """Return total number of engrams."""
        cursor = await self._conn_or_raise.execute("SELECT COUNT(*) FROM engrams")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def learn(self, engram: Engram) -> Engram:
        """Store a new engram or update existing (dedup by content_hash).

        Args:
            engram: The engram to store.

        Returns:
            The stored engram (possibly with bumped access_count).
        """
        conn = self._conn_or_raise
        now_ms = int(time.time() * 1000)

        cursor = await conn.execute(
            "SELECT id, access_count, retrieval_strength FROM engrams WHERE content_hash = ?",
            (engram.content_hash,),
        )
        row = await cursor.fetchone()

        if row:
            new_access = row["access_count"] + 1
            new_strength = min(row["retrieval_strength"] + 0.1, 1.0)
            await conn.execute(
                """UPDATE engrams
                   SET access_count = ?, retrieval_strength = ?, last_accessed_at = ?
                   WHERE id = ?""",
                (new_access, new_strength, now_ms, row["id"]),
            )
            await conn.commit()
            found = await self._get_by_id(conn, row["id"])
            assert found is not None
            return found

        engram_id = engram.id or str(uuid.uuid4())
        tags_json = json.dumps(engram.tags, ensure_ascii=False)
        signals_json = json.dumps(engram.feedback_signals, ensure_ascii=False)
        cb_json = json.dumps(engram.circuit_breaker) if engram.circuit_breaker else None
        embedding_blob: bytes | None = None
        if engram.embedding:
            import numpy as np
            embedding_blob = np.array(engram.embedding, dtype=np.float32).tobytes()
        associations_json = json.dumps(engram.associations)

        await conn.execute(
            """INSERT INTO engrams (
                id, statement, content_hash, domain, agent, source,
                retrieval_strength, created_at, last_accessed_at, access_count,
                last_feedback_at, commitment, pinned, tags, feedback_signals,
                circuit_breaker, associations_str, fact_key, fact_value,
                valid_from, valid_to, embedding, embedding_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                engram_id,
                engram.statement,
                engram.content_hash,
                engram.domain,
                engram.agent,
                engram.source,
                engram.retrieval_strength,
                engram.created_at,
                engram.last_accessed_at,
                engram.access_count,
                engram.last_feedback_at,
                engram.commitment,
                1 if engram.pinned else 0,
                tags_json,
                signals_json,
                cb_json,
                associations_json,
                engram.fact_key,
                engram.fact_value,
                engram.valid_from,
                engram.valid_to,
                embedding_blob,
                engram.embedding_model,
            ),
        )
        await conn.commit()
        found = await self._get_by_id(conn, engram_id)
        assert found is not None
        return found

    async def recall(self, engram_id: str) -> Engram | None:
        """Retrieve an engram by ID and bump its access count.

        Args:
            engram_id: The engram ID.

        Returns:
            The Engram, or None if not found.
        """
        conn = self._conn_or_raise
        engram = await self._get_by_id(conn, engram_id)
        if engram is None:
            return None
        now_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE engrams SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
            (now_ms, engram_id),
        )
        await conn.commit()
        return await self._get_by_id(conn, engram_id)

    async def feedback(self, engram_id: str, signal: str) -> Engram:
        """Apply a feedback signal to an engram.

        Args:
            engram_id: The engram ID.
            signal: 'positive', 'negative', or 'neutral'.

        Returns:
            The updated Engram.

        Raises:
            KeyError: If the engram does not exist.
        """
        conn = self._conn_or_raise
        cursor = await conn.execute("SELECT * FROM engrams WHERE id = ?", (engram_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Engram not found: {engram_id}")

        signals = json.loads(row["feedback_signals"])
        signals[signal] = signals.get(signal, 0) + 1

        cb = json.loads(row["circuit_breaker"]) if row["circuit_breaker"] else None
        new_strength = float(row["retrieval_strength"])
        new_commitment = str(row["commitment"])
        now_ms = int(time.time() * 1000)

        if signal == "negative":
            new_strength = max(new_strength * 0.5, 0.0)
            cb = cb or {"failures": 0, "locked_until": 0}
            cb["failures"] += 1
            if cb["failures"] >= 3:
                cb["locked_until"] = now_ms + 3_600_000
                new_commitment = "exploring"

        elif signal == "positive":
            new_strength = min(new_strength + 0.3, 1.0)
            commitment_order = {"exploring": 0, "leaning": 1, "decided": 2, "locked": 3}
            if commitment_order.get(new_commitment, 0) < 2:
                if new_commitment == "exploring":
                    new_commitment = "leaning"
                elif new_commitment == "leaning":
                    new_commitment = "decided"

        await conn.execute(
            """UPDATE engrams
               SET feedback_signals = ?, retrieval_strength = ?,
                   circuit_breaker = ?, last_feedback_at = ?, commitment = ?
               WHERE id = ?""",
            (
                json.dumps(signals),
                new_strength,
                json.dumps(cb) if cb else None,
                now_ms,
                new_commitment,
                engram_id,
            ),
        )
        await conn.commit()
        updated = await self._get_by_id(conn, engram_id)
        assert updated is not None
        return updated

    async def forget(self, engram_id: str) -> None:
        """Delete an engram by ID.

        Args:
            engram_id: The engram ID to delete.
        """
        conn = self._conn_or_raise
        await conn.execute("DELETE FROM engrams WHERE id = ?", (engram_id,))
        await conn.commit()

    async def consolidate(self, merge_threshold: float = 0.95) -> dict[str, Any]:
        """Consolidate engrams: merge similar, prune weak ones (placeholder).

        Args:
            merge_threshold: Cosine similarity threshold for merging.
        """
        return {"merged": 0, "pruned": 0}

    async def _get_by_id(self, conn: aiosqlite.Connection, engram_id: str) -> Engram | None:
        """Read an engram from the database and convert to Engram model."""
        cursor = await conn.execute("SELECT * FROM engrams WHERE id = ?", (engram_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_engram(row)

    @staticmethod
    def _row_to_engram(row: aiosqlite.Row) -> Engram:
        """Convert a SQLite Row to an Engram model."""
        embedding: list[float] | None = None
        if row["embedding"] is not None:
            import numpy as np
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()

        return Engram(
            id=row["id"],
            statement=row["statement"],
            content_hash=row["content_hash"],
            domain=row["domain"],
            agent=row["agent"],
            source=row["source"],
            retrieval_strength=row["retrieval_strength"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            last_feedback_at=row["last_feedback_at"],
            commitment=row["commitment"],
            pinned=bool(row["pinned"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            feedback_signals=json.loads(row["feedback_signals"]) if row["feedback_signals"] else {},
            circuit_breaker=json.loads(row["circuit_breaker"]) if row["circuit_breaker"] else None,
            associations=json.loads(row["associations_str"]) if row["associations_str"] else [],
            fact_key=row["fact_key"],
            fact_value=row["fact_value"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            embedding=embedding,
            embedding_model=row["embedding_model"],
        )
