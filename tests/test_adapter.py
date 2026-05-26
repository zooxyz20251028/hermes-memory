"""Tests for hermes-memory → Hermes integration adapter."""

from __future__ import annotations

import os
import tempfile

import pytest

from hermes_memory.adapters import sync_learn, sync_search


class TestAdapter:
    """Synchronous adapter wrapper tests."""

    def test_sync_learn_and_search(self):
        """sync_learn should store and sync_search should find."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "adapter_test.db")
            os.environ["HERMES_MEMORY_DB"] = db_path

            # Store a memory
            result = sync_learn(
                "适配器测试: sync_learn 写入成功",
                domain="project",
                commitment="decided",
            )
            assert result["action"] == "created"
            assert result["commitment"] == "decided"

            # Search for it
            results = sync_search("适配器测试")
            assert len(results) > 0
            assert any("sync_learn" in r["statement"] for r in results)

    def test_sync_learn_dedup(self):
        """Repeating same statement should dedup with access_count bump."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "adapter_dedup.db")
            os.environ["HERMES_MEMORY_DB"] = db_path

            r1 = sync_learn("去重测试语句", domain="project")
            r2 = sync_learn("去重测试语句", domain="project")
            assert r1["action"] == "created"
            assert r2["action"] == "updated"
            assert r2["access_count"] > r1["access_count"]
