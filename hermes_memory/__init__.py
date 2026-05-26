"""Hermes Memory — multi-source hybrid memory system for AI agents.

Phase 1 (done): models, schema, embedding, store — 31 tests
Phase 2-3 (done): searcher (4-path + RRF + BM25), compressor (AAAK-style)
Phase 4 (done): analyzer (LLM-driven session learning, replaces memory_learner.py)

Adapted from:
  - MemPalace (MIT): searcher.py BM25+hybrid rank, dialect.py AAAK compression
  - OpenSpace (Apache 2.0): analyzer.py ExecutionAnalyzer pattern
  - Plur (MIT): Engram model with ACT-R activation, selectAndSpread
  - GBrain: RRF fusion design, dual-track format
"""

from hermes_memory.models import Engram
from hermes_memory.schema import get_all_ddl
from hermes_memory.store import MemoryStore
from hermes_memory.embedding import EmbeddingClient
from hermes_memory.searcher import Searcher
from hermes_memory.compressor import Compressor
from hermes_memory.analyzer import Analyzer, SessionContext

__all__ = [
    "Engram",
    "get_all_ddl",
    "MemoryStore",
    "EmbeddingClient",
    "Searcher",
    "Compressor",
    "Analyzer",
    "SessionContext",
]
