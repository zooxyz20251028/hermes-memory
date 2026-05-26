"""
Multi-path hybrid search with RRF fusion.

Adapted from MemPalace searcher.py (MIT) — its BM25 implementation,
hybrid ranking strategy, and FTS5 fallback patterns are battle-tested.
Extended with RRF (Reciprocal Rank Fusion) from GBrain's design and
association-graph propagation for memory-specific retrieval.

4 retrieval paths:
  1. FTS5 BM25 full-text search (exact term matching)
  2. Vector cosine similarity (semantic proximity via BaiLian embedding-v4)
  3. Association graph (1-hop propagation via engram associations)
  4. RRF fusion — score = Σ 1/(k + rank_i) across active paths

Integration points:
  - MemoryStore (store.py) — for engram CRUD
  - EmbeddingClient (embedding.py) — for vector generation + cosine sim
  - Engram (models.py) — frozen data model with associations

Usage:
  searcher = Searcher(store, embed_client)
  results = await searcher.search("用户偏好 DeepSeek", top_k=10)
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite
    from hermes_memory.models import Engram
    from hermes_memory.store import MemoryStore
    from hermes_memory.embedding import EmbeddingClient

# ── RRF constants ──
RRF_K: int = 60  # RRF smoothing constant (standard)

# ── BM25 constants (from MemPalace) ──
BM25_K1: float = 1.5  # term-frequency saturation
BM25_B: float = 0.75  # length normalization

# ── Association depth ──
MAX_GRAPH_HOPS: int = 1  # max propagation depth
MAX_GRAPH_NODES: int = 20  # max nodes returned from graph path


class Searcher:
    """Multi-path search engine with RRF fusion.

    Args:
        store: Initialized MemoryStore.
        embed_client: Initialized EmbeddingClient.
    """

    def __init__(self, store: MemoryStore, embed_client: EmbeddingClient) -> None:
        self._store = store
        self._embed = embed_client

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        paths: tuple[str, ...] = ("fts5", "vector", "graph"),
    ) -> list[Engram]:
        """Search engrams via configured paths, fuse with RRF.

        Args:
            query: Search query (Chinese or English).
            top_k: Number of results to return.
            paths: Active retrieval paths. Default: all three.

        Returns:
            List of Engram models, ordered by descending RRF score.
        """
        if not query.strip():
            return []

        path_results: dict[str, list[tuple[Engram, float]]] = {}
        tasks: list[asyncio.Task[None]] = []

        async def _run_path(name: str, coro) -> None:
            results = await coro
            if results:
                path_results[name] = results

        if "fts5" in paths:
            tasks.append(asyncio.create_task(
                _run_path("fts5", self._fts5_search(query, top_k * 3)),
            ))
        if "vector" in paths:
            tasks.append(asyncio.create_task(
                _run_path("vector", self._vector_search(query, top_k * 3)),
            ))
        if "graph" in paths:
            tasks.append(asyncio.create_task(
                _run_path("graph", self._graph_search(query, top_k * 3)),
            ))

        if tasks:
            await asyncio.gather(*tasks)

        if not path_results:
            return []

        return await self._rrf_fuse(path_results, top_k)

    # ═══════════════════════════════════════════════════════════════════
    # Path 1: FTS5 BM25 full-text search
    # ═══════════════════════════════════════════════════════════════════

    async def _fts5_search(self, query: str, top_k: int) -> list[tuple[Engram, float]]:
        """FTS5 full-text search with Okapi-BM25 scoring.

        Uses SQLite FTS5 external-content table (engrams_fts) that mirrors
        the `engrams` table.  Falls back to LIKE-based search when FTS5
        produces no results (the Wiki RAG FTS5 search degradation strategy,
        adapted for engram memory).
        """
        conn = self._store._conn_or_raise
        tokens = self._tokenize(query)

        # ── Layer 1: FTS5 phrase match ──
        results = await self._fts5_phrase(conn, query, tokens, top_k)
        if results:
            return self._bm25_rerank(query, results, top_k)

        # ── Layer 2: FTS5 AND match ──
        if len(tokens) >= 2:
            results = await self._fts5_and(conn, tokens, top_k)
            if results:
                return self._bm25_rerank(query, results, top_k)

        # ── Layer 3: FTS5 OR match ──
        results = await self._fts5_or(conn, tokens, top_k)
        if results:
            return self._bm25_rerank(query, results, top_k)

        # ── Layer 4: LIKE fallback ──
        results = await self._like_fallback(conn, query, tokens, top_k)
        return self._bm25_rerank(query, results, top_k)

    async def _fts5_phrase(
        self, conn: "aiosqlite.Connection", query: str, tokens: list[str], top_k: int,
    ) -> list[tuple[Engram, float]]:
        """Exact phrase match — highest precision."""
        escaped = query.replace('"', '""')
        try:
            cursor = await conn.execute(
                """SELECT e.id, e.statement, e.domain, e.commitment, e.tags
                   FROM engrams_fts f
                   JOIN engrams e ON e.rowid = f.rowid
                   WHERE engrams_fts MATCH ?
                   LIMIT ?""",
                (f'"{escaped}"', top_k),
            )
            rows = await cursor.fetchall()
            return [(await self._store._get_by_id(conn, r["id"]), 9999.0) for r in rows if r["id"]]
        except Exception:
            return []

    async def _fts5_and(
        self, conn: "aiosqlite.Connection", tokens: list[str], top_k: int,
    ) -> list[tuple[Engram, float]]:
        """AND conjunction of all tokens."""
        fts_query = " AND ".join(tokens)
        try:
            cursor = await conn.execute(
                """SELECT e.id, e.statement, e.domain, e.commitment, e.tags
                   FROM engrams_fts f
                   JOIN engrams e ON e.rowid = f.rowid
                   WHERE engrams_fts MATCH ?
                   LIMIT ?""",
                (fts_query, top_k),
            )
            rows = await cursor.fetchall()
            return [(await self._store._get_by_id(conn, r["id"]), 500.0) for r in rows if r["id"]]
        except Exception:
            return []

    async def _fts5_or(
        self, conn: "aiosqlite.Connection", tokens: list[str], top_k: int,
    ) -> list[tuple[Engram, float]]:
        """OR disjunction of all tokens."""
        fts_query = " OR ".join(tokens)
        try:
            cursor = await conn.execute(
                """SELECT e.id, e.statement, e.domain, e.commitment, e.tags
                   FROM engrams_fts f
                   JOIN engrams e ON e.rowid = f.rowid
                   WHERE engrams_fts MATCH ?
                   LIMIT ?""",
                (fts_query, top_k),
            )
            rows = await cursor.fetchall()
            return [(await self._store._get_by_id(conn, r["id"]), 400.0) for r in rows if r["id"]]
        except Exception:
            return []

    async def _like_fallback(
        self, conn: "aiosqlite.Connection", query: str, tokens: list[str], top_k: int,
    ) -> list[tuple[Engram, float]]:
        """LIKE-based fallback when FTS5 returns nothing."""
        terms = [q.strip() for q in query.split() if len(q.strip()) >= 2]
        if not terms:
            terms = tokens

        like_clauses = " OR ".join(["e.statement LIKE ?"] * len(terms))
        like_params = [f"%{t}%" for t in terms]

        cursor = await conn.execute(
            f"""SELECT e.id, e.statement, e.domain, e.commitment, e.tags
                FROM engrams e
                WHERE {like_clauses}
                LIMIT ?""",
            (*like_params, top_k),
        )
        rows = await cursor.fetchall()
        return [(await self._store._get_by_id(conn, r["id"]), 200.0) for r in rows if r["id"]]

    # ═══════════════════════════════════════════════════════════════════
    # Path 2: Vector semantic search
    # ═══════════════════════════════════════════════════════════════════

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[Engram, float]]:
        """Cosine similarity search over engram embeddings.

        Generates query vector via EmbeddingClient, then compares against
        all stored embeddings with cosine similarity.  Uses numpy for
        vectorized computation when many engrams exist.
        """
        query_vec = await self._embed.get_embedding(query)
        if not query_vec:
            return []

        conn = self._store._conn_or_raise
        cursor = await conn.execute(
            "SELECT id, statement, embedding FROM engrams WHERE embedding IS NOT NULL",
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        scores: list[tuple[tuple, float]] = []
        for row in rows:
            emb = row["embedding"]
            if emb is None:
                continue
            import numpy as np
            stored_vec = np.frombuffer(emb, dtype=np.float32)
            sim = self._embed.cosine_similarity(query_vec, stored_vec)
            scores.append((row, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        results: list[tuple[Engram, float]] = []
        for row, sim in top:
            engram = await self._store._get_by_id(conn, row["id"])
            if engram:
                results.append((engram, sim))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Path 3: Association graph propagation
    # ═══════════════════════════════════════════════════════════════════

    async def _graph_search(self, query: str, top_k: int) -> list[tuple[Engram, float]]:
        """1-hop association graph search.

        Identifies seed engrams (matches query tokens), then follows
        their association links to discover related engrams.  Supports
        multi-hop propagation up to MAX_GRAPH_HOPS.
        """
        conn = self._store._conn_or_raise
        tokens = self._tokenize(query)

        # Find seed engrams by token overlap
        cursor = await conn.execute(
            """SELECT e.id, e.statement, e.associations_str, e.domain
               FROM engrams e
               WHERE e.associations_str != '[]'""",
        )
        rows = await cursor.fetchall()

        import json
        seen: set[str] = set()
        results: list[tuple[Engram, float]] = []

        # Score seeds by token overlap
        for row in rows:
            stmt = row["statement"]
            overlap = sum(1 for t in tokens if t in stmt.lower())
            if overlap > 0:
                engram = await self._store._get_by_id(conn, row["id"])
                if engram and engram.id not in seen:
                    seen.add(engram.id)
                    results.append((engram, overlap * 0.25))
                    # Follow associations
                    assoc_ids = json.loads(row["associations_str"]) if row["associations_str"] else []
                    for aid in assoc_ids[:MAX_GRAPH_HOPS * 3]:
                        if aid not in seen:
                            assoc_engram = await self._store._get_by_id(conn, aid)
                            if assoc_engram:
                                seen.add(aid)
                                results.append((assoc_engram, overlap * 0.15))
                                if len(results) >= MAX_GRAPH_NODES:
                                    break

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # RRF Fusion (Reciprocal Rank Fusion)
    # ═══════════════════════════════════════════════════════════════════

    async def _rrf_fuse(
        self,
        path_results: dict[str, list[tuple[Engram, float]]],
        top_k: int,
    ) -> list[Engram]:
        """Fuse multiple ranked lists via Reciprocal Rank Fusion.

        RRF score = Σ 1/(k + rank_i) for each path i where the engram
        appears.  k = RRF_K (60) smooths the impact of rank variation.

        This is preferred over weighted fusion because:
          - No normalization needed across paths with different score ranges
          - Robust to path failures (missing paths don't zero out scores)
          - Naturally handles ties (multiple paths contributing)
        """
        rrf_scores: dict[str, float] = {}

        for _path_name, ranked_list in path_results.items():
            for rank, (engram, _) in enumerate(ranked_list):
                eid = engram.id
                rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score descending
        ranked: list[tuple[str, float]] = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True,
        )

        # Resolve to Engram models (batch lookup by IDs)
        conn = self._store._conn_or_raise
        results: list[Engram] = []
        for eid, _score in ranked[:top_k]:
            engram = await self._store._get_by_id(conn, eid)
            if engram is not None:
                results.append(engram)

        return results

    # ═══════════════════════════════════════════════════════════════════
    # BM25 Re-ranking (from MemPalace searcher.py)
    # ═══════════════════════════════════════════════════════════════════

    def _bm25_rerank(
        self,
        query: str,
        candidates: list[tuple[Engram, float]],
        top_k: int,
    ) -> list[tuple[Engram, float]]:
        """Re-rank candidates with Okapi-BM25 scoring.

        IDF is computed over the candidate set (corpus-relative),
        following MemPalace's design: IDF reflects how discriminative
        each query term is *within the candidates*.

        Args:
            query: Original search query.
            candidates: List of (engram, initial_score) tuples.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of (engram, bm25_score) tuples.
        """
        if not candidates:
            return []

        # Compute BM25 over the candidate corpus
        docs = [engram.statement for engram, _ in candidates]
        query_terms = set(self._tokenize(query))

        if not query_terms or not docs:
            return candidates[:top_k]

        n_docs = len(docs)
        tokenized = [self._tokenize(d) for d in docs]
        doc_lens = [len(toks) for toks in tokenized]
        avgdl = sum(doc_lens) / n_docs if n_docs else 1.0

        # Document frequency
        df: dict[str, int] = {term: 0 for term in query_terms}
        for toks in tokenized:
            seen = set(toks) & query_terms
            for term in seen:
                df[term] += 1

        # Smooth IDF: log((N - df + 0.5) / (df + 0.5) + 1)  — from MemPalace
        idf: dict[str, float] = {
            term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
            for term in query_terms
        }

        # BM25 scoring (k1=1.5, b=0.75)
        scored: list[tuple[Engram, float]] = []
        for i, (engram, _) in enumerate(candidates):
            dl = doc_lens[i]
            if dl == 0:
                scored.append((engram, 0.0))
                continue
            tf: dict[str, int] = {}
            for t in tokenized[i]:
                if t in query_terms:
                    tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for term, freq in tf.items():
                num = freq * (BM25_K1 + 1)
                den = freq + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
                score += idf[term] * num / den
            scored.append((engram, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # Tokenizer (from MemPalace searcher.py)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase + alphanumeric tokens of length ≥ 2.

        For CJK text, individual characters are treated as tokens.
        For alphabetic text, consecutive word characters form tokens.
        """
        import re
        if not text:
            return []

        # CJK range
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        # Word tokens
        words = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())

        return cjk + words
