"""Hermes Agent integration adapter for hermes-memory.

Plugs hermes-memory MCP tools into Hermes' existing tool system
via the standard MCP server integration (config.yaml).

Setup:
  1. Add to ~/.hermes/config.yaml:
     mcp_servers:
       memory:
         command: python3
         args: [-m, hermes_memory.mcp_server]
         cwd: ~/.hermes/scripts/hermes_memory/
         env:
           HERMES_MEMORY_DB: ~/.hermes/scripts/hermes_memory/data/memory.db
           DASHSCOPE_API_KEY: sk-xxx

  2. Restart Hermes gateway. Tools appear as:
     - hermes_memory_learn, hermes_memory_search, hermes_memory_context, etc.

  3. Replace old memory calls:
     Old: memory(action='add', target='memory', content='...')
     New: hermes_memory_learn(statement='...', domain='project', commitment='decided')

This module also provides a synchronous wrapper for scripts:
  from hermes_memory.adapters.hermes_adapter import sync_learn, sync_search
  sync_learn("fact here", domain="project")
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def sync_learn(
    statement: str,
    domain: str = "general",
    commitment: str = "leaning",
    tags: str = "[]",
) -> dict[str, Any]:
    """Synchronous wrapper: store a memory via hermes-memory CLI.

    For use in cron jobs and scripts where async MCP is overkill.

    Args:
        statement: Memory content.
        domain: user|project|reference|workflow|general.
        commitment: locked|decided|leaning|exploring.
        tags: JSON list string.

    Returns:
        Result dict with id, statement, commitment, access_count, action.
    """
    db_path = os.environ.get(
        "HERMES_MEMORY_DB",
        str(Path.home() / ".hermes" / "scripts" / "hermes_memory" / "data" / "memory.db"),
    )

    async def _do():
        from hermes_memory.store import MemoryStore
        from hermes_memory.models import Engram
        import hashlib, time

        store = MemoryStore(db_path)
        await store.initialize()

        now_ms = int(time.time() * 1000)
        eng = Engram(
            statement=statement.strip(),
            content_hash=hashlib.sha256(statement.strip().encode()).hexdigest(),
            domain=domain, agent="hermes", source="memory_tool",
            created_at=now_ms, last_accessed_at=now_ms,
            commitment=commitment, tags=json.loads(tags),
        )
        result = await store.learn(eng)
        await store.close()
        return {
            "id": result.id,
            "statement": result.statement,
            "commitment": result.commitment,
            "access_count": result.access_count,
            "action": "updated" if result.access_count > 1 else "created",
        }

    return asyncio.run(_do())


def sync_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Synchronous wrapper: search memory via hermes-memory CLI.

    For use in cron jobs and scripts.

    Args:
        query: Search query.
        top_k: Max results.

    Returns:
        List of result dicts.
    """
    db_path = os.environ.get(
        "HERMES_MEMORY_DB",
        str(Path.home() / ".hermes" / "scripts" / "hermes_memory" / "data" / "memory.db"),
    )

    async def _do():
        from hermes_memory.store import MemoryStore
        from hermes_memory.searcher import Searcher
        from hermes_memory.embedding import EmbeddingClient

        store = MemoryStore(db_path)
        await store.initialize()
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        embed = EmbeddingClient(api_key=api_key) if api_key else EmbeddingClient(api_key="no-key")
        searcher = Searcher(store, embed)
        # Skip vector if no real API key
        paths = ("fts5", "graph")
        if api_key:
            paths = ("fts5", "vector", "graph")
        results = await searcher.search(query, top_k=top_k, paths=paths)
        await store.close()
        return [
            {"id": e.id, "statement": e.statement, "domain": e.domain, "commitment": e.commitment}
            for e in results
        ]

    return asyncio.run(_do())


def sync_context(query: str, max_tokens: int = 2000) -> str:
    """Synchronous wrapper: build injection context.

    For replacing build_context.py in the session bootstrap.

    Args:
        query: Current task description.
        max_tokens: Token budget.

    Returns:
        Compressed context string.
    """
    db_path = os.environ.get(
        "HERMES_MEMORY_DB",
        str(Path.home() / ".hermes" / "scripts" / "hermes_memory" / "data" / "memory.db"),
    )

    async def _do():
        from hermes_memory.store import MemoryStore
        from hermes_memory.searcher import Searcher
        from hermes_memory.compressor import Compressor
        from hermes_memory.injector import Injector
        from hermes_memory.embedding import EmbeddingClient

        store = MemoryStore(db_path)
        await store.initialize()
        embed = EmbeddingClient(api_key=os.environ.get("DASHSCOPE_API_KEY", ""))
        searcher = Searcher(store, embed)
        compressor = Compressor()
        injector = Injector(store, searcher, compressor, embed)
        ctx = await injector.build_context(query, max_tokens=max_tokens)
        await store.close()
        return ctx

    return asyncio.run(_do())
