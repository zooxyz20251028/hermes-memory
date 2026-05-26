"""
Health scoring system — ported from GBrain Health Score (V2 plan §2.9).

Computes a 4-dimensional health score for the memory system:
  1. Embedding coverage (30 pts): % of engrams with embeddings
  2. Association density (25 pts): % of engrams with associations
  3. Circuit breaker rate (20 pts): % of engrams NOT in circuit-break
  4. Freshness (25 pts): % of engrams accessed in last 30 days

This makes memory health quantifiable — no more "black box rot".
Target: ≥80 (healthy), <50 (needs Dream Cycle).

ECC: Pure read-only computation. No mutations.

Usage:
  from hermes_memory.health import HealthChecker
  checker = HealthChecker(store)
  score = await checker.compute()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_memory.store import MemoryStore


# ── Scoring weights ──
EMBED_WEIGHT: float = 30.0  # embedding coverage
ASSOC_WEIGHT: float = 25.0  # association density
CB_WEIGHT: float = 20.0     # circuit-breaker health
FRESH_WEIGHT: float = 25.0  # freshness (30-day window)

# ── Thresholds ──
HEALTHY_THRESHOLD: float = 80.0
WARNING_THRESHOLD: float = 60.0
CRITICAL_THRESHOLD: float = 40.0

# ── Freshness window ──
FRESHNESS_WINDOW_MS: int = 30 * 86400 * 1000  # 30 days in ms

# ── Default top-K for recommendations ──
TOP_STALE: int = 5


class HealthChecker:
    """Compute 4-dimensional memory system health score.

    Args:
        store: Initialized MemoryStore.
    """

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    async def compute(self) -> dict:
        """Compute full health report.

        Returns:
            Dict with 'score', 'status', and per-dimension breakdown.
        """
        conn = self._store._conn_or_raise

        cursor = await conn.execute("SELECT COUNT(*) as total FROM engrams")
        total = (await cursor.fetchone())[0]
        if total == 0:
            return {
                "score": 100.0,
                "status": "healthy",
                "total": 0,
                "embedded": 0,
                "linked": 0,
                "broken": 0,
                "fresh": 0,
                "dimensions": {},
            }

        # 1. Embedding coverage
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams WHERE embedding IS NOT NULL",
        )
        embedded = (await cursor.fetchone())[0]

        # 2. Association density
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams WHERE associations_str != '[]'",
        )
        linked = (await cursor.fetchone())[0]

        # 3. Circuit-breaker health (not broken)
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams",
        )
        total_2 = (await cursor.fetchone())[0]
        broken = total_2 - (await self._count_not_broken(conn))

        # 4. Freshness
        import time
        now_ms = int(time.time() * 1000)
        fresh_cutoff = now_ms - FRESHNESS_WINDOW_MS
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM engrams WHERE last_accessed_at >= ?",
            (fresh_cutoff,),
        )
        fresh = (await cursor.fetchone())[0]

        # Compute scores
        embed_score = (embedded / total) * EMBED_WEIGHT
        assoc_score = (linked / total) * ASSOC_WEIGHT
        cb_score = max(0, (1 - (broken / total))) * CB_WEIGHT
        fresh_score = (fresh / total) * FRESH_WEIGHT

        total_score = round(embed_score + assoc_score + cb_score + fresh_score, 1)

        # Status
        if total_score >= HEALTHY_THRESHOLD:
            status = "healthy"
        elif total_score >= WARNING_THRESHOLD:
            status = "warning"
        elif total_score >= CRITICAL_THRESHOLD:
            status = "degraded"
        else:
            status = "critical"

        return {
            "score": total_score,
            "status": status,
            "total": total,
            "embedded": embedded,
            "linked": linked,
            "broken": broken,
            "fresh": fresh,
            "dimensions": {
                "embedding_coverage": round(embed_score, 1),
                "association_density": round(assoc_score, 1),
                "circuit_breaker_health": round(cb_score, 1),
                "freshness": round(fresh_score, 1),
            },
            "recommendations": self._recommendations(status, embedded, linked, fresh, total),
        }

    async def _count_not_broken(self, conn) -> int:
        """Count engrams NOT in circuit-break state."""
        cursor = await conn.execute("SELECT id, circuit_breaker FROM engrams")
        import json
        import time
        rows = await cursor.fetchall()
        not_broken = 0
        now_ms = int(time.time() * 1000)
        for row in rows:
            cb = row[1]
            if cb is None:
                not_broken += 1
                continue
            try:
                cb_dict = json.loads(cb)
                locked_until = cb_dict.get("locked_until", 0)
                if locked_until < now_ms:
                    not_broken += 1
            except (json.JSONDecodeError, TypeError):
                not_broken += 1
        return not_broken

    @staticmethod
    def _recommendations(
        status: str,
        embedded: int,
        linked: int,
        fresh: int,
        total: int,
    ) -> list[str]:
        """Generate human-readable recommendations based on score."""
        recs: list[str] = []
        if total == 0:
            return ["No engrams yet — run session_learn or add manually."]

        if embedded / total < 0.5:
            recs.append(f"Low embedding coverage ({embedded}/{total}) — run batch embedding.")
        if linked / total < 0.3:
            recs.append(f"Low association density ({linked}/{total}) — run Dream Cycle cross-ref.")
        if fresh / total < 0.3:
            recs.append(f"Low freshness ({fresh}/{total}) — many stale engrams; run Dream Cycle prune.")
        if status == "critical":
            recs.append("CRITICAL: Run Dream Cycle immediately — 7-phase consolidation needed.")
        elif status == "degraded":
            recs.append("Memory health degraded — schedule Dream Cycle at next maintenance window.")

        return recs or ["All dimensions healthy. 🟢"]
