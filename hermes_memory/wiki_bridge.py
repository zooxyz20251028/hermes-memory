"""
Wiki knowledge-base bridge — search and write integration.

Bridges the hermes-memory system with the wiki-rag knowledge base
(``~/wiki/`` markdown files indexed by ``wiki-rag`` into SQLite).

Two directions:
  1. SEARCH: Query wiki-rag's FTS5 + vector index from memory searcher.
     Wiki results participate in RRF fusion alongside engram results.
  2. WRITE:  Auto-create wiki concept pages when analyzer extracts
     high-confidence durable facts (confidence ≥ 0.8).

Design (ECC-compliant):
  - Immutable: ``WikiSearchResult`` is frozen (like Engram).
  - No side effects on read; write operations are idempotent.
  - AGENTS.md rules enforced: frontmatter, indexing, logging.

Usage:
  bridge = WikiBridge()
  results = bridge.search("报价公式", top_k=5)
  bridge.write_concept("标题", "内容", tags=["标签"], confidence=0.9)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# ── Paths ──
_WIKI_ROOT: Path = Path.home() / "wiki"
_WIKI_RAG_DB: Path = Path.home() / ".hermes" / "scripts" / "wiki-rag" / "data" / "wiki_rag.db"
_WIKI_INDEX: Path = _WIKI_ROOT / "索引.md"
_WIKI_LOG: Path = _WIKI_ROOT / "日志.md"

# ── Concept page template (AGENTS.md rule ②) ──
_CONCEPT_FRONTMATTER = """---
title: {title}
created: {date}
updated: {date}
type: concept
tags: {tags}
confidence: {confidence}
sources: [auto-extracted]
---

"""

# ── AGENTS.md rule ⑩: confidence tiers ──
TIER1_THRESHOLD: float = 0.8  # ≥0.8 → instant write
TIER2_THRESHOLD: float = 0.6  # 0.6-0.79 → queued for cron


@dataclass(frozen=True)
class WikiSearchResult:
    """Frozen result from wiki-rag search (immutable, like Engram)."""

    doc_id: int
    title: str
    excerpt: str
    rel_path: str
    score: float
    tags: tuple[str, ...] = ()


class WikiBridge:
    """Bridge between hermes-memory and wiki-rag knowledge base.

    Args:
        wiki_root: Path to wiki markdown directory.
        db_path: Path to wiki-rag SQLite database.
    """

    def __init__(
        self,
        wiki_root: str | None = None,
        db_path: str | None = None,
    ) -> None:
        self._wiki_root = Path(wiki_root) if wiki_root else _WIKI_ROOT
        self._db_path = Path(db_path) if db_path else _WIKI_RAG_DB

    # ═════════════════════════════════════════════════════════════════
    # SEARCH — query wiki-rag FTS5 + LIKE fallback
    # ═════════════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 5) -> list[WikiSearchResult]:
        """Search wiki-rag knowledge base.

        4-layer degradation (from Wiki RAG FTS5 strategy):
          1. FTS5 phrase match (exact) → score 9999
          2. FTS5 AND match → score 500
          3. FTS5 OR match → score 400
          4. LIKE fallback → score 200

        Args:
            query: Search query string.
            top_k: Max results to return.

        Returns:
            List of frozen WikiSearchResult objects.
        """
        if not query.strip() or not self._db_path.exists():
            return []

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            tokens = _tokenize(query)

            # Layer 1: phrase match
            results = self._fts5_phrase(conn, query, top_k)
            if results:
                return results

            # Layer 2: AND
            if len(tokens) >= 2:
                results = self._fts5_and(conn, tokens, top_k)
                if results:
                    return results

            # Layer 3: OR
            results = self._fts5_or(conn, tokens, top_k)
            if results:
                return results

            # Layer 4: LIKE
            return self._like_fallback(conn, query, top_k)

        finally:
            conn.close()

    def _fts5_phrase(
        self, conn: sqlite3.Connection, query: str, top_k: int,
    ) -> list[WikiSearchResult]:
        escaped = query.replace('"', '""')
        try:
            rows = conn.execute(
                """SELECT f.rowid, d.title, d.content, d.rel_path, d.tags
                   FROM wiki_fts_index f
                   JOIN wiki_docs d ON d.doc_id = f.rowid
                   WHERE wiki_fts_index MATCH ?
                   LIMIT ?""",
                (f'"{escaped}"', top_k),
            ).fetchall()
            return [
                WikiSearchResult(
                    doc_id=r[0], title=r[1], excerpt=_excerpt(r[2], query, 100),
                    rel_path=r[3], score=9999.0, tags=tuple(json.loads(r[4] or "[]")),
                )
                for r in rows
            ]
        except sqlite3.Error:
            return []

    def _fts5_and(
        self, conn: sqlite3.Connection, tokens: list[str], top_k: int,
    ) -> list[WikiSearchResult]:
        fts_q = " AND ".join(tokens)
        try:
            rows = conn.execute(
                """SELECT f.rowid, d.title, d.content, d.rel_path, d.tags
                   FROM wiki_fts_index f
                   JOIN wiki_docs d ON d.doc_id = f.rowid
                   WHERE wiki_fts_index MATCH ?
                   LIMIT ?""",
                (fts_q, top_k),
            ).fetchall()
            return [
                WikiSearchResult(
                    doc_id=r[0], title=r[1], excerpt=_excerpt(r[2], " ".join(tokens), 100),
                    rel_path=r[3], score=500.0, tags=tuple(json.loads(r[4] or "[]")),
                )
                for r in rows
            ]
        except sqlite3.Error:
            return []

    def _fts5_or(
        self, conn: sqlite3.Connection, tokens: list[str], top_k: int,
    ) -> list[WikiSearchResult]:
        fts_q = " OR ".join(tokens)
        try:
            rows = conn.execute(
                """SELECT f.rowid, d.title, d.content, d.rel_path, d.tags
                   FROM wiki_fts_index f
                   JOIN wiki_docs d ON d.doc_id = f.rowid
                   WHERE wiki_fts_index MATCH ?
                   LIMIT ?""",
                (fts_q, top_k),
            ).fetchall()
            return [
                WikiSearchResult(
                    doc_id=r[0], title=r[1], excerpt=_excerpt(r[2], " ".join(tokens), 100),
                    rel_path=r[3], score=400.0, tags=tuple(json.loads(r[4] or "[]")),
                )
                for r in rows
            ]
        except sqlite3.Error:
            return []

    def _like_fallback(
        self, conn: sqlite3.Connection, query: str, top_k: int,
    ) -> list[WikiSearchResult]:
        terms = [q.strip() for q in query.split() if len(q.strip()) >= 2]
        if not terms:
            return []
        clauses = " OR ".join(["d.title LIKE ? OR d.content LIKE ?"] * len(terms))
        params: list[str] = []
        for t in terms:
            params.extend([f"%{t}%", f"%{t}%"])
        rows = conn.execute(
            f"""SELECT d.doc_id, d.title, d.content, d.rel_path, d.tags
                FROM wiki_docs d
                WHERE {clauses}
                LIMIT ?""",
            (*params, top_k),
        ).fetchall()
        return [
            WikiSearchResult(
                doc_id=r[0], title=r[1], excerpt=_excerpt(r[2], query, 100),
                rel_path=r[3], score=200.0, tags=tuple(json.loads(r[4] or "[]")),
            )
            for r in rows
        ]

    # ═════════════════════════════════════════════════════════════════
    # WRITE — auto-create concept pages (AGENTS.md rule ⑩)
    # ═════════════════════════════════════════════════════════════════

    def write_concept(
        self,
        title: str,
        content: str,
        *,
        tags: Sequence[str] = (),
        confidence: float = 0.8,
    ) -> str | None:
        """Auto-create a wiki concept page (Tier 1: confidence ≥ 0.8).

        Follows AGENTS.md rules ② (frontmatter), ③ (index update), and
        log convention.  Idempotent: skips if same title already exists.

        Args:
            title: Page title (becomes filename: ``概念/{safe_title}.md``).
            content: Markdown body content.
            tags: 1-3 topic tags.
            confidence: Extraction confidence (0.0-1.0).

        Returns:
            Relative path of created file, or None if skipped.
        """
        # Confidence gate
        if confidence < TIER1_THRESHOLD:
            return None

        safe_name = _sanitise_filename(title)
        filepath = self._wiki_root / "概念" / f"{safe_name}.md"

        # Idempotent: skip if exists
        if filepath.exists():
            return None

        # Write
        date_str = time.strftime("%Y-%m-%d")
        tags_json = json.dumps(list(tags)[:3], ensure_ascii=False)
        body = _CONCEPT_FRONTMATTER.format(
            title=title, date=date_str, tags=tags_json, confidence=confidence,
        )
        # Generate a quick summary from the statement
        summary = f"## 一句话总结\n\n{content[:200].strip()}\n\n"
        detail = f"## 要点\n\n{content.strip()}\n\n"
        auto_note = "\n---\n\n*本页面由 hermes-memory WikiBridge 自动生成*\n"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(body + summary + detail + auto_note, encoding="utf-8")

        # Update index + log (AGENTS.md rule ③)
        self._update_index()
        self._append_log(date_str, f"auto | 概念/{safe_name}.md — {title}")

        return f"概念/{safe_name}.md"

    def queue_for_cron(self, title: str, content: str, tags: Sequence[str] = ()) -> None:
        """Queue a Tier-2 candidate for cron batch processing.

        Writes to a JSON queue file that wiki_extract_decisions.py reads
        at 00:05 daily.

        Args:
            title: Page title.
            content: Page content.
            tags: Topic tags.
        """
        queue_path = Path.home() / ".hermes" / "scripts" / "wiki_queue.json"
        entry = {
            "title": title,
            "content": content,
            "tags": list(tags),
            "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        existing: list[dict] = []
        if queue_path.exists():
            try:
                existing = json.loads(queue_path.read_text())
            except json.JSONDecodeError:
                existing = []
        # Deduplicate by title
        if not any(e.get("title") == title for e in existing):
            existing.append(entry)
            queue_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    # ═════════════════════════════════════════════════════════════════
    # Index maintenance (AGENTS.md rule ③)
    # ═════════════════════════════════════════════════════════════════

    def _update_index(self) -> None:
        """Update the concept page count in 索引.md."""
        if not _WIKI_INDEX.exists():
            return
        concept_dir = self._wiki_root / "概念"
        count = len(list(concept_dir.glob("*.md"))) if concept_dir.exists() else 0
        content = _WIKI_INDEX.read_text(encoding="utf-8")
        # Update the total pages line
        pattern = r"概念\d+"
        replacement = f"概念{count}"
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            _WIKI_INDEX.write_text(new_content, encoding="utf-8")

    def _append_log(self, date_str: str, entry: str) -> None:
        """Append a line to 日志.md."""
        if not _WIKI_LOG.exists():
            return
        log_content = _WIKI_LOG.read_text(encoding="utf-8")
        # Insert after the most recent date header, or at the top
        date_header = f"## [{date_str}]"
        if date_header in log_content:
            insert_pos = log_content.index(date_header) + len(date_header)
            # Find end of that day's block
            next_header = log_content.find("\n## [", insert_pos)
            if next_header > insert_pos:
                log_content = (
                    log_content[:next_header]
                    + f"\n- {entry}"
                    + log_content[next_header:]
                )
            else:
                log_content += f"\n- {entry}\n"
        else:
            # New date: insert at top
            log_content = f"{date_header}\n- {entry}\n\n{log_content}"
        _WIKI_LOG.write_text(log_content, encoding="utf-8")


# ── Helpers ──

def _tokenize(text: str) -> list[str]:
    """Tokenize for FTS5: CJK chars + alphanumeric words ≥ 2 chars."""
    if not text:
        return []
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
    return cjk + words


def _excerpt(content: str, query: str, max_chars: int = 100) -> str:
    """Extract a relevant excerpt around query terms."""
    if not content:
        return ""
    # Find first occurrence of any query term
    terms = query.split()
    best_pos = 0
    for term in terms:
        pos = content.lower().find(term.lower())
        if pos >= 0:
            best_pos = pos
            break
    start = max(0, best_pos - max_chars // 2)
    end = min(len(content), start + max_chars)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(content):
        snippet += "…"
    return snippet


def _sanitise_filename(title: str) -> str:
    """Convert title to safe filename: no special chars, max 64 chars."""
    safe = re.sub(r"[^\w\-\u4e00-\u9fff]", "-", title)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe[:64] if safe else "untitled"
