"""Hermes Memory — multi-source hybrid memory system for AI agents.

Adapted from production-grade source code (not lightweight rewrites):
  - MemPalace (MIT): searcher.py BM25+hybrid rank, dialect.py AAAK compression
  - OpenSpace (Apache 2.0): analyzer.py ExecutionAnalyzer pattern, evolver.py Dream Cycle
  - Plur (MIT): Engram model with ACT-R activation, selectAndSpread, feedback + circuit breaker
  - GBrain: RRF fusion design, classifyQuery intent, health score, token budget

Phase 1: models/schema/embedding/store — 31 tests
Phase 2: searcher — 5-path FTS5 BM25 + vector + graph + wiki + RRF — 8 tests
Phase 3: compressor — AAAK-style compression — 6 tests
Phase 4: analyzer — LLM-driven session learning — 6 tests
Phase 5: wiki_bridge — cross-island wiki-rag search + write fusion — 7 tests
Phase 6: decay + intention + token_budget + health — 14 tests
Phase 7: injector (selectAndSpread V2) + dream_cycle (7-phase) — 12 tests

Total: 84 tests, all green.
"""

from hermes_memory.models import Engram
from hermes_memory.schema import get_all_ddl
from hermes_memory.store import MemoryStore
from hermes_memory.embedding import EmbeddingClient
from hermes_memory.searcher import Searcher
from hermes_memory.compressor import Compressor
from hermes_memory.analyzer import Analyzer, SessionContext
from hermes_memory.wiki_bridge import WikiBridge, WikiSearchResult
from hermes_memory.decay import compute_retrieval_strength, predict_strength_after_days, is_stale
from hermes_memory.intention import classify_query, QueryIntent, IntentType
from hermes_memory.token_budget import select_for_injection, estimate_tokens, budget_stats
from hermes_memory.health import HealthChecker
from hermes_memory.injector import Injector
from hermes_memory.dream_cycle import DreamCycle

__all__ = [
    "Engram",
    "get_all_ddl",
    "MemoryStore",
    "EmbeddingClient",
    "Searcher",
    "Compressor",
    "Analyzer",
    "SessionContext",
    "WikiBridge",
    "WikiSearchResult",
    "classify_query",
    "QueryIntent",
    "IntentType",
    "compute_retrieval_strength",
    "predict_strength_after_days",
    "is_stale",
    "select_for_injection",
    "estimate_tokens",
    "budget_stats",
    "HealthChecker",
    "Injector",
    "DreamCycle",
]
