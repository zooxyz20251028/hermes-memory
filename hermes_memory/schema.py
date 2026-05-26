"""SQLite schema definitions — engrams, associations, facts, FTS5 tables.

Immutability: all constants are frozen sets/tuples.
DDL statements are split so each can be executed individually.
"""

from __future__ import annotations


CREATE_ENGRAMS_TABLE = """CREATE TABLE IF NOT EXISTS engrams (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL DEFAULT 'general',
    agent TEXT NOT NULL DEFAULT 'hermes',
    source TEXT NOT NULL DEFAULT 'memory_tool',
    retrieval_strength REAL NOT NULL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 1,
    last_feedback_at INTEGER,
    commitment TEXT NOT NULL DEFAULT 'exploring',
    pinned INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]',
    feedback_signals TEXT NOT NULL DEFAULT '{}',
    circuit_breaker TEXT,
    associations_str TEXT NOT NULL DEFAULT '[]',
    fact_key TEXT,
    fact_value TEXT,
    valid_from INTEGER,
    valid_to INTEGER,
    embedding BLOB,
    embedding_model TEXT
);"""

CREATE_FTS5_TABLE = """CREATE VIRTUAL TABLE IF NOT EXISTS engrams_fts USING fts5(
    statement, tags, domain,
    content='engrams',
    content_rowid='rowid',
    tokenize='unicode61'
);"""

CREATE_ASSOCIATIONS_TABLE = """CREATE TABLE IF NOT EXISTS associations (
    source_id TEXT NOT NULL REFERENCES engrams(id),
    target_id TEXT NOT NULL REFERENCES engrams(id),
    strength REAL NOT NULL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (source_id, target_id)
);"""

CREATE_FACTS_TABLE = """CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    engram_id TEXT NOT NULL REFERENCES engrams(id),
    metric TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT,
    period TEXT,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER,
    created_at INTEGER NOT NULL
);"""

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_engrams_domain ON engrams(domain);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_agent ON engrams(agent);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_commitment ON engrams(commitment);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_pinned ON engrams(pinned);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_content_hash ON engrams(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_retrieval_strength ON engrams(retrieval_strength);",
    "CREATE INDEX IF NOT EXISTS idx_engrams_created_at ON engrams(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_associations_source ON associations(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_associations_target ON associations(target_id);",
    "CREATE INDEX IF NOT EXISTS idx_facts_engram ON facts(engram_id);",
    "CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric);",
)


def get_all_ddl() -> list[str]:
    """Return all DDL statements in dependency order, one per item."""
    return [
        CREATE_ENGRAMS_TABLE,
        CREATE_FTS5_TABLE,
        CREATE_ASSOCIATIONS_TABLE,
        CREATE_FACTS_TABLE,
        *INDEX_STATEMENTS,
    ]
