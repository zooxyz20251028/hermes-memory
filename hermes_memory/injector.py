"""
Select-and-Spread V2 injection pipeline — the core context builder.

Orchestrates multi-path retrieval, RRF fusion, decay computation,
commitment-weighted token-budget selection, and formatted output for
LLM context injection.  This is the main entry point that replaces
the old ``build_context.py`` / ``memory_v4.py inject`` pipeline.

Pipeline (from V2 plan §2.5):
  1. Classify query intent → adjust path weights
  2. Multi-path recall: FTS5 BM25 + vector + graph + wiki (parallel)
  3. RRF fusion → merged ranked list
  4. Post-processing: commitment boost, ACT-R decay penalty, recency penalty, feedback penalty
  5. Token budget execution → select top engrams under token cap
  6. Format → compressed context string for system prompt injection

Adapted from Plur's selectAndSpread (MIT) with GBrain's token budget,
and extended with Wiki-rag cross-knowledge-base fusion.

ECC:  Orchestration layer — no direct mutations.  All operations are
      read-only on MemoryStore.  Formatter is a pure function.

Usage:
  injector = Injector(store, searcher, compressor, embed_client, wiki_bridge)
  context = await injector.build_context("用户查询报价公式", max_tokens=2000)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_memory.models import Engram
    from hermes_memory.store import MemoryStore
    from hermes_memory.searcher import Searcher
    from hermes_memory.compressor import Compressor
    from hermes_memory.embedding import EmbeddingClient

# ── Post-processing weights ──
COMMITMENT_BOOST: dict[str, float] = {
    "locked": 1.5,
    "decided": 1.2,
    "leaning": 1.0,
    "exploring": 0.8,
}
RECENCY_PENALTY_DAYS: int = 90       # days after which recency penalty applies
RECENCY_PENALTY_FACTOR: float = 0.5  # multiply score by this when > 90 days old
FEEDBACK_PENALTY_THRESHOLD: int = 3  # negative > 3 × positive → penalty
FEEDBACK_PENALTY_FACTOR: float = 0.3  # multiply score by this when penalty applies


class Injector:
    """Select-and-Spread V2 context injection orchestrator.

    Args:
        store: MemoryStore for reading engrams.
        searcher: Searcher with multi-path + RRF fusion.
        compressor: Compressor for engram → compact string.
        embed_client: EmbeddingClient for vector search.
        wiki_bridge: Optional WikiBridge for cross-KB search.
    """

    def __init__(
        self,
        store: "MemoryStore",
        searcher: "Searcher",
        compressor: "Compressor",
        embed_client: "EmbeddingClient",
        wiki_bridge=None,
    ) -> None:
        self._store = store
        self._searcher = searcher
        self._compressor = compressor
        self._embed = embed_client
        self._wiki = wiki_bridge

    async def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        *,
        paths: tuple[str, ...] = ("fts5", "vector", "graph", "wiki"),
    ) -> str:
        """Build injection context for an LLM session.

        Full selectAndSpread V2 pipeline: intent → search → RRF →
        post-process → budget → format.

        Args:
            query: Natural-language description of current context/task.
            max_tokens: Maximum token budget for the context block.
            paths: Active search paths.

        Returns:
            Formatted context string ready for system prompt injection.
        """
        # 1. Classify intent → adjust search weights
        from hermes_memory.intention import classify_query
        intent = classify_query(query)

        # 2. Multi-path recall (parallel)
        engrams = await self._searcher.search(query, top_k=20, paths=paths)

        # 3. RRF fusion already done by searcher — engrams are ranked
        if not engrams:
            return ""

        # 4. Post-process: apply commitment/decay/recency/feedback modifiers
        scored = self._post_process(engrams)

        # 5. Token budget selection
        from hermes_memory.token_budget import select_for_injection
        selected = select_for_injection(scored, max_tokens=max_tokens)

        # 6. Format
        return self._compressor.compress_batch(selected, max_tokens=max_tokens)

    async def quick_context(self, query: str) -> str:
        """Fast injection (1000 tokens) for latency-sensitive use."""
        return await self.build_context(query, max_tokens=1000)

    async def deep_context(self, query: str) -> str:
        """Comprehensive injection (4000 tokens) for deep recall."""
        return await self.build_context(query, max_tokens=4000)

    # ── Post-processing ──

    def _post_process(self, engrams: list["Engram"]) -> list["Engram"]:
        """Apply commitment boost + decay + recency + feedback modifiers.

        Returns engrams sorted by adjusted importance (no mutations).
        """
        from hermes_memory.decay import compute_retrieval_strength

        now_ms = int(time.time() * 1000)
        scored: list[tuple[float, "Engram"]] = []

        for eng in engrams:
            score = 1.0

            # Commitment boost
            score *= COMMITMENT_BOOST.get(eng.commitment, 1.0)

            # ACT-R + Weibull decay penalty
            decay_strength = compute_retrieval_strength(eng, now_ms)
            score *= decay_strength

            # Recency penalty (> 90 days old)
            age_ms = now_ms - eng.created_at
            age_days = age_ms / 86400000
            if age_days > RECENCY_PENALTY_DAYS:
                score *= RECENCY_PENALTY_FACTOR

            # Feedback penalty (negative > 3 × positive)
            pos = eng.feedback_signals.get("positive", 0)
            neg = eng.feedback_signals.get("negative", 0)
            if neg > FEEDBACK_PENALTY_THRESHOLD * max(pos, 1):
                score *= FEEDBACK_PENALTY_FACTOR

            scored.append((score, eng))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [eng for _, eng in scored]
