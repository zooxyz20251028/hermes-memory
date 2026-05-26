"""
Hermes-memory MCP server — JSON-RPC over stdio.

Exposes 10 memory tools via the MCP protocol.  Uses raw JSON-RPC
over stdin/stdout (like MemPalace's mcp_server.py, 2869 lines proven
in production) rather than the unstable MCP Python SDK.

Protocol: one JSON-RPC request per line on stdin, one response on stdout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from hermes_memory.models import Engram
from hermes_memory.store import MemoryStore
from hermes_memory.embedding import EmbeddingClient
from hermes_memory.searcher import Searcher
from hermes_memory.compressor import Compressor
from hermes_memory.analyzer import Analyzer
from hermes_memory.wiki_bridge import WikiBridge
from hermes_memory.injector import Injector
from hermes_memory.dream_cycle import DreamCycle
from hermes_memory.health import HealthChecker

# ── Config ──
DB_PATH: str = os.environ.get(
    "HERMES_MEMORY_DB",
    str(Path.home() / ".hermes" / "scripts" / "hermes_memory" / "data" / "memory.db"),
)
EMBED_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")

# ── Global state ──
_store: MemoryStore | None = None
_searcher: Searcher | None = None
_compressor: Compressor | None = None
_wiki: WikiBridge | None = None
_injector: Injector | None = None
_dreamer: DreamCycle | None = None


async def _init() -> MemoryStore:
    global _store, _searcher, _compressor, _wiki, _injector, _dreamer
    if _store is not None:
        return _store

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _store = MemoryStore(DB_PATH)
    await _store.initialize()

    embed = EmbeddingClient(api_key=EMBED_API_KEY)
    _wiki = WikiBridge()
    _searcher = Searcher(_store, embed, wiki_bridge=_wiki)
    _compressor = Compressor()
    _injector = Injector(_store, _searcher, _compressor, embed, wiki_bridge=_wiki)
    _dreamer = DreamCycle(_store, embed)

    return _store


# ── Tool definitions ──
TOOLS = [
    {
        "name": "hermes_memory_learn",
        "description": "Store a new memory engram or update existing (dedup by content hash).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "Memory content to store"},
                "domain": {"type": "string", "enum": list(Engram.VALID_DOMAINS)},
                "commitment": {"type": "string", "enum": list(Engram.VALID_COMMITMENTS)},
                "tags": {"type": "string", "description": "JSON list, e.g. '[\"报价\",\"规则\"]'"},
                "fact_key": {"type": "string"},
                "fact_value": {"type": "string"},
            },
            "required": ["statement"],
        },
    },
    {
        "name": "hermes_memory_recall",
        "description": "Retrieve a memory engram by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"engram_id": {"type": "string"}},
            "required": ["engram_id"],
        },
    },
    {
        "name": "hermes_memory_search",
        "description": "Multi-path hybrid search with RRF fusion (FTS5 BM25 + vector + graph + wiki).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10},
                "paths": {"type": "string", "default": "fts5,vector,graph,wiki"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hermes_memory_feedback",
        "description": "Apply positive/negative/neutral feedback to a memory engram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "engram_id": {"type": "string"},
                "signal": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            },
            "required": ["engram_id", "signal"],
        },
    },
    {
        "name": "hermes_memory_forget",
        "description": "Delete a memory engram permanently.",
        "inputSchema": {
            "type": "object",
            "properties": {"engram_id": {"type": "string"}},
            "required": ["engram_id"],
        },
    },
    {
        "name": "hermes_memory_health",
        "description": "Compute 4-dim health score (embedding coverage, association density, circuit-breaker, freshness).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hermes_memory_context",
        "description": "Build injection context via selectAndSpread V2 pipeline for LLM system prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Current context/task description"},
                "max_tokens": {"type": "integer", "default": 2000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hermes_memory_dream",
        "description": "Trigger 7-phase dream cycle for nightly memory consolidation.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hermes_memory_wiki",
        "description": "Cross-search the wiki knowledge base (FTS5 + LIKE fallback).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hermes_memory_stats",
        "description": "Return memory system statistics (count, domain/commitment breakdowns).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ── Tool dispatch ──

async def _dispatch(name: str, args: dict) -> str:
    store = await _init()

    if name == "hermes_memory_learn":
        return await _cmd_learn(store, args)
    if name == "hermes_memory_recall":
        return await _cmd_recall(store, args)
    if name == "hermes_memory_search":
        return await _cmd_search(args)
    if name == "hermes_memory_feedback":
        return await _cmd_feedback(store, args)
    if name == "hermes_memory_forget":
        return await _cmd_forget(store, args)
    if name == "hermes_memory_health":
        return await _cmd_health(store)
    if name == "hermes_memory_context":
        return await _cmd_context(args)
    if name == "hermes_memory_dream":
        return await _cmd_dream()
    if name == "hermes_memory_wiki":
        return _cmd_wiki(args)
    if name == "hermes_memory_stats":
        return await _cmd_stats(store)

    return json.dumps({"error": f"Unknown tool: {name}"})


async def _cmd_learn(store, args):
    stmt = args["statement"].strip()
    domain = args.get("domain", "general")
    commitment = args.get("commitment", "leaning")
    tags_str = args.get("tags", "[]")
    now_ms = int(time.time() * 1000)
    content_hash = hashlib.sha256(stmt.encode()).hexdigest()
    try:
        parsed_tags = json.loads(tags_str)
    except (json.JSONDecodeError, TypeError):
        parsed_tags = []
    eng = Engram(
        statement=stmt, content_hash=content_hash,
        domain=domain, agent="hermes", source="memory_tool",
        created_at=now_ms, last_accessed_at=now_ms,
        commitment=commitment, tags=parsed_tags[:5],
        fact_key=args.get("fact_key"), fact_value=args.get("fact_value"),
    )
    result = await store.learn(eng)
    return json.dumps({"id": result.id, "statement": result.statement, "commitment": result.commitment, "access_count": result.access_count, "action": "updated" if result.access_count > 1 else "created"}, ensure_ascii=False)


async def _cmd_recall(store, args):
    eng = await store.recall(args["engram_id"])
    if eng is None:
        return json.dumps({"error": f"Not found: {args['engram_id']}"})
    return json.dumps({"id": eng.id, "statement": eng.statement, "domain": eng.domain, "commitment": eng.commitment, "tags": eng.tags, "retrieval_strength": eng.retrieval_strength, "access_count": eng.access_count}, ensure_ascii=False)


async def _cmd_search(args):
    searcher = _searcher
    paths = tuple(p.strip() for p in args.get("paths", "fts5,vector,graph,wiki").split(",") if p.strip())
    results = await searcher.search(args["query"], top_k=args.get("top_k", 10), paths=paths)
    items = [{"rank": i + 1, "id": e.id, "statement": e.statement, "domain": e.domain, "commitment": e.commitment, "tags": e.tags} for i, e in enumerate(results)]
    return json.dumps({"query": args["query"], "results": items}, ensure_ascii=False)


async def _cmd_feedback(store, args):
    try:
        eng = await store.feedback(args["engram_id"], args["signal"])
        return json.dumps({"id": eng.id, "commitment": eng.commitment, "retrieval_strength": eng.retrieval_strength, "feedback_signals": eng.feedback_signals}, ensure_ascii=False)
    except KeyError:
        return json.dumps({"error": f"Not found: {args['engram_id']}"})


async def _cmd_forget(store, args):
    await store.forget(args["engram_id"])
    return json.dumps({"status": "deleted", "id": args["engram_id"]})


async def _cmd_health(store):
    checker = HealthChecker(store)
    report = await checker.compute()
    return json.dumps(report, ensure_ascii=False)


async def _cmd_context(args):
    injector = _injector
    ctx = await injector.build_context(args["query"], max_tokens=args.get("max_tokens", 2000))
    return ctx if ctx else "No relevant memories found."


async def _cmd_dream():
    dreamer = _dreamer
    report = await dreamer.run()
    return json.dumps(report, ensure_ascii=False)


def _cmd_wiki(args):
    wiki = _wiki or WikiBridge()
    results = wiki.search(args["query"], top_k=args.get("top_k", 5))
    items = [{"rank": i + 1, "title": r.title, "excerpt": r.excerpt, "path": r.rel_path, "score": r.score, "tags": list(r.tags)} for i, r in enumerate(results)]
    return json.dumps({"query": args["query"], "results": items}, ensure_ascii=False)


async def _cmd_stats(store):
    conn = store._conn_or_raise
    cur = await conn.execute("SELECT COUNT(*) FROM engrams")
    total = (await cur.fetchone())[0]
    cur = await conn.execute("SELECT domain, COUNT(*) FROM engrams GROUP BY domain")
    domains = {row[0]: row[1] async for row in cur}
    cur = await conn.execute("SELECT commitment, COUNT(*) FROM engrams GROUP BY commitment")
    commitments = {row[0]: row[1] async for row in cur}
    cur = await conn.execute("SELECT COUNT(*) FROM associations")
    assoc = (await cur.fetchone())[0]
    cur = await conn.execute("SELECT COUNT(*) FROM facts")
    facts = (await cur.fetchone())[0]
    return json.dumps({"total_engrams": total, "domains": domains, "commitments": commitments, "associations": assoc, "facts": facts}, ensure_ascii=False)


# ── JSON-RPC loop (pattern from MemPalace mcp_server.py §2811-2869) ──

async def _run() -> None:
    """MCP stdio server loop."""
    # Write init message to stderr
    print("hermes-memory MCP server starting...", file=sys.stderr)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        method = request.get("method", "")

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hermes-memory", "version": "0.5.0"},
                },
            }
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": TOOLS},
            }
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            text = await _dispatch(tool_name, tool_args)
            resp = {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        elif method == "notifications/initialized":
            # Silently acknowledge
            continue
        else:
            resp = {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    print("hermes-memory MCP server stopped.", file=sys.stderr)


def main() -> None:
    """Entry point."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
