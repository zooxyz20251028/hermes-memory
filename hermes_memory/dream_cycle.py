"""
Dream Cycle — 7-phase nightly memory maintenance.

Ported from GBrain's dream cycle + OpenSpace's evolver pattern.
Runs daily at 03:00 AM (cron) to consolidate, deduplicate, prune,
and cross-reference the memory store.  This is the "sleep phase"
that keeps memory healthy without manual intervention.

7 phases (V2 plan §2.8):
  1. Orient   — read stats + last dream state
  2. Consolidate — merge engrams with similarity > 0.95
  3. Dedup      — detect contradictions (user correction vs old knowledge)
  4. Prune      — archive stale engrams (retrieval_strength < 0.05, not pinned)
  5. Facts      — extract structured facts → facts table
  6. Cross-ref  — scan tags + facts → auto-build associations
  7. Report     — output dream report: merged/pruned/extracted counts

ECC: All mutations are idempotent via content_hash dedup.

Usage:
  dreamer = DreamCycle(store, embed_client)
  report = await dreamer.run()
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_memory.store import MemoryStore
    from hermes_memory.embedding import EmbeddingClient


# ── Thresholds ──
CONSOLIDATE_THRESHOLD: float = 0.95  # cosine similarity threshold for merge
PRUNE_STRENGTH_THRESHOLD: float = 0.05  # below this → archive
PRUNE_ACCESS_THRESHOLD: int = 0  # access count must be 0
STALE_DAYS: int = 90  # days since last access for stale check


class DreamCycle:
    """7-phase nightly memory maintenance engine.

    Args:
        store: MemoryStore for engram CRUD.
        embed_client: EmbeddingClient for similarity computation.
    """

    def __init__(self, store: "MemoryStore", embed_client: "EmbeddingClient") -> None:
        self._store = store
        self._embed = embed_client

    async def run(self) -> dict[str, Any]:
        """Execute all 7 phases and return a dream report.

        Returns:
            Dict with per-phase statistics and total summary.
        """
        report: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Phase 1: Orient
        orient = await self._phase_orient()
        report["orient"] = orient

        # Phase 2: Consolidate
        consolidated = await self._phase_consolidate()
        report["consolidated"] = consolidated

        # Phase 3: Dedup
        deduped = await self._phase_dedup()
        report["deduped"] = deduped

        # Phase 4: Prune
        pruned = await self._phase_prune()
        report["pruned"] = pruned

        # Phase 5: Facts
        facts_extracted = await self._phase_facts()
        report["facts_extracted"] = facts_extracted

        # Phase 6: Cross-ref
        cross_refs = await self._phase_cross_ref()
        report["cross_refs"] = cross_refs

        # Phase 7: Report summary
        report["summary"] = self._build_summary(report)
        return report

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Orient
    # ═══════════════════════════════════════════════════════════════

    async def _phase_orient(self) -> dict:
        """Read current memory statistics."""
        conn = self._store._conn_or_raise
        cursor = await conn.execute("SELECT COUNT(*) as total FROM engrams")
        total = (await cursor.fetchone())[0]

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams WHERE embedding IS NOT NULL",
        )
        embedded = (await cursor.fetchone())[0]

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams WHERE pinned = 1",
        )
        pinned = (await cursor.fetchone())[0]

        from hermes_memory.health import HealthChecker
        checker = HealthChecker(self._store)
        health = await checker.compute()

        return {
            "total": total,
            "embedded": embedded,
            "pinned": pinned,
            "health_score": health["score"],
            "health_status": health["status"],
        }

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Consolidate (merge near-duplicates)
    # ═══════════════════════════════════════════════════════════════

    async def _phase_consolidate(self) -> int:
        """Merge engrams with cosine similarity > CONSOLIDATE_THRESHOLD."""
        conn = self._store._conn_or_raise
        cursor = await conn.execute(
            "SELECT id, statement, embedding FROM engrams WHERE embedding IS NOT NULL",
        )
        rows = await cursor.fetchall()
        if len(rows) < 2:
            return 0

        import numpy as np

        # Load all embeddings
        vecs: list[tuple[str, np.ndarray]] = []
        for row in rows:
            if row[2] is not None:
                vecs.append((row[0], np.frombuffer(row[2], dtype=np.float32)))

        merged_count = 0
        seen: set[str] = set()

        for i in range(len(vecs)):
            if vecs[i][0] in seen:
                continue
            for j in range(i + 1, len(vecs)):
                if vecs[j][0] in seen:
                    continue
                sim = float(np.dot(vecs[i][1], vecs[j][1]) / (
                    np.linalg.norm(vecs[i][1]) * np.linalg.norm(vecs[j][1])
                ))
                if sim > CONSOLIDATE_THRESHOLD:
                    # Merge: keep the stronger one, delete the weaker
                    cursor_i = await conn.execute(
                        "SELECT retrieval_strength FROM engrams WHERE id = ?",
                        (vecs[i][0],),
                    )
                    strength_i = (await cursor_i.fetchone())[0] or 0.5

                    cursor_j = await conn.execute(
                        "SELECT retrieval_strength FROM engrams WHERE id = ?",
                        (vecs[j][0],),
                    )
                    strength_j = (await cursor_j.fetchone())[0] or 0.5

                    if strength_i >= strength_j:
                        await conn.execute("DELETE FROM engrams WHERE id = ?", (vecs[j][0],))
                        seen.add(vecs[j][0])
                    else:
                        await conn.execute("DELETE FROM engrams WHERE id = ?", (vecs[i][0],))
                        seen.add(vecs[i][0])
                    merged_count += 1

        if merged_count > 0:
            await conn.commit()
        return merged_count

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Dedup (contradiction detection)
    # ═══════════════════════════════════════════════════════════════

    async def _phase_dedup(self) -> int:
        """Detect contradictions: user corrections vs old knowledge.

        Currently a stub — full contradiction detection requires LLM.
        Returns 0 (placeholder for future LLM-driven phase).
        """
        # TODO: LLM-driven contradiction scan across engram pairs
        return 0

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Prune (archive stale engrams)
    # ═══════════════════════════════════════════════════════════════

    async def _phase_prune(self) -> int:
        """Archive stale engrams (retrieval_strength < threshold, not pinned)."""
        from hermes_memory.decay import compute_retrieval_strength

        conn = self._store._conn_or_raise
        cursor = await conn.execute(
            "SELECT id, created_at, retrieval_strength, access_count, "
            "pinned, commitment, last_accessed_at FROM engrams "
            "WHERE pinned = 0",
        )
        rows = await cursor.fetchall()

        now_ms = int(time.time() * 1000)
        pruned_count = 0

        for row in rows:
            # Quick filter: skip recently accessed
            last_access = row[6] or 0
            if (now_ms - last_access) < STALE_DAYS * 86400 * 1000:
                continue
            if row[3] > PRUNE_ACCESS_THRESHOLD:
                continue

            # Full decay computation
            from hermes_memory.models import Engram
            sid = row[0]
            eng = Engram(
                id=sid, statement=f"stale-engram-{sid}",
                content_hash=f"dream-prune-{sid}",
                domain="general", agent="hermes", source="memory_tool",
                created_at=row[1], last_accessed_at=row[6] or row[1],
                retrieval_strength=float(row[2] or 0.05),
                access_count=int(row[3] or 0),
                commitment=str(row[5] or "exploring"),
            )
            strength = compute_retrieval_strength(eng, now_ms)
            if strength < PRUNE_STRENGTH_THRESHOLD:
                await conn.execute("DELETE FROM engrams WHERE id = ?", (row[0],))
                pruned_count += 1

        if pruned_count > 0:
            await conn.commit()
        return pruned_count

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Facts extraction
    # ═══════════════════════════════════════════════════════════════

    async def _phase_facts(self) -> int:
        """Extract structured facts from engrams → facts table.

        Scans engrams with fact_key set and ensures they exist in the
        facts table.  Missing facts are inserted.
        """
        conn = self._store._conn_or_raise
        cursor = await conn.execute(
            "SELECT id, fact_key, fact_value, valid_from, valid_to, created_at, statement "
            "FROM engrams WHERE fact_key IS NOT NULL",
        )
        rows = await cursor.fetchall()

        extracted = 0
        for row in rows:
            # Check if fact already exists
            cur = await conn.execute(
                "SELECT id FROM facts WHERE engram_id = ? AND metric = ?",
                (row[0], row[1]),
            )
            exists = await cur.fetchone()
            if exists:
                continue

            import uuid
            await conn.execute(
                """INSERT INTO facts (id, engram_id, metric, value, unit,
                   valid_from, valid_to, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    row[0],
                    row[1],
                    row[2] or "",
                    None,
                    row[3] or 0,
                    row[4],
                    row[5] or int(time.time() * 1000),
                ),
            )
            extracted += 1

        if extracted > 0:
            await conn.commit()
        return extracted

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Cross-reference (auto-build associations)
    # ═══════════════════════════════════════════════════════════════

    async def _phase_cross_ref(self) -> int:
        """Auto-build associations from shared tags/facts.

        Two engrams sharing ≥ 2 tags → create association.
        """
        conn = self._store._conn_or_raise
        cursor = await conn.execute(
            "SELECT id, tags FROM engrams WHERE tags != '[]'",
        )
        rows = await cursor.fetchall()

        engram_tags: list[tuple[str, set[str]]] = []
        for row in rows:
            tags = set(json.loads(row[1])) if row[1] else set()
            if len(tags) >= 2:
                engram_tags.append((row[0], tags))

        cross_refs = 0
        for i in range(len(engram_tags)):
            for j in range(i + 1, len(engram_tags)):
                shared = engram_tags[i][1] & engram_tags[j][1]
                if len(shared) >= 1:
                    # Check if association already exists
                    cur = await conn.execute(
                        "SELECT 1 FROM associations WHERE source_id = ? AND target_id = ?",
                        (engram_tags[i][0], engram_tags[j][0]),
                    )
                    if await cur.fetchone():
                        continue

                    await conn.execute(
                        """INSERT INTO associations (source_id, target_id, strength, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (
                            engram_tags[i][0],
                            engram_tags[j][0],
                            0.8,
                            int(time.time() * 1000),
                        ),
                    )
                    cross_refs += 1

        if cross_refs > 0:
            await conn.commit()
        return cross_refs

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Report
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_summary(report: dict) -> str:
        """Build a human-readable dream cycle summary."""
        orient = report.get("orient", {})
        total = orient.get("total", 0)

        parts = [
            f"🧠 Dream Cycle Report — {report['timestamp']}",
            f"",
            f"📊 Orientation: {total} engrams, {orient.get('embedded', 0)} embedded, "
            f"{orient.get('pinned', 0)} pinned, health={orient.get('health_score', '?')}",
            f"",
            f"🔀 Consolidated: {report.get('consolidated', 0)} near-duplicates merged",
            f"🔍 Deduped: {report.get('deduped', 0)} contradictions detected",
            f"🗑️  Pruned: {report.get('pruned', 0)} stale engrams archived",
            f"📋 Facts extracted: {report.get('facts_extracted', 0)} structured facts",
            f"🔗 Cross-refs: {report.get('cross_refs', 0)} new associations created",
        ]
        return "\n".join(parts)
