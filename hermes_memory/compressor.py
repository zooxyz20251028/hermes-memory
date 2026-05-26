"""
Engram-to-compact-string compression for context injection.

Adapted from MemPalace dialect.py (MIT) — AAAK compression format
extracts entities, topics, key sentences, and flags.  This module
adapts the algorithm for engram memory: takes ``Engram`` objects
and produces compact context strings suitable for injection into
the LLM system prompt with minimal token overhead.

Key changes from MemPalace:
  - Works on single engrams, not files
  - Adds commitment-level prefix (🔒 locked, ✅ decided, 🤔 leaning, 🔍 exploring)
  - Preserves fact_key/fact_value for structured facts
  - Supports Chinese text with CJK-specific tokenization
  - Output format: ``commitment_icon [domain] tags: statement (flags)``

Usage:
  compressor = Compressor()
  compact = compressor.compress(engram)
  # => "✅ [user] DeepSeek偏好: 马兴堂偏爱使用DeepSeek V4 Flash (DECISION+CORE)"
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_memory.models import Engram

# ── Commitment-level icons ──
_COMMITMENT_ICONS: dict[str, str] = {
    "locked": "🔒",
    "decided": "✅",
    "leaning": "🤔",
    "exploring": "🔍",
}

# ── Flag signals (from MemPalace dialect.py) ──
_FLAG_SIGNALS: dict[str, str] = {
    "decided": "DECISION",
    "chose": "DECISION",
    "switched": "DECISION",
    "migrated": "DECISION",
    "replaced": "DECISION",
    "instead of": "DECISION",
    "because": "DECISION",
    "决定": "DECISION",
    "选择": "DECISION",
    "迁移": "DECISION",
    "替换": "DECISION",
    "founded": "ORIGIN",
    "created": "ORIGIN",
    "started": "ORIGIN",
    "born": "ORIGIN",
    "launched": "ORIGIN",
    "first time": "ORIGIN",
    "创建": "ORIGIN",
    "初始": "ORIGIN",
    "core": "CORE",
    "fundamental": "CORE",
    "essential": "CORE",
    "principle": "CORE",
    "belief": "CORE",
    "always": "CORE",
    "never forget": "CORE",
    "核心": "CORE",
    "根本": "CORE",
    "无论如何": "CORE",
    "turning point": "PIVOT",
    "changed everything": "PIVOT",
    "realized": "PIVOT",
    "breakthrough": "PIVOT",
    "epiphany": "PIVOT",
    "转折": "PIVOT",
    "突破": "PIVOT",
    "api": "TECHNICAL",
    "database": "TECHNICAL",
    "architecture": "TECHNICAL",
    "deploy": "TECHNICAL",
    "infrastructure": "TECHNICAL",
    "algorithm": "TECHNICAL",
    "framework": "TECHNICAL",
    "server": "TECHNICAL",
    "config": "TECHNICAL",
    "配置": "TECHNICAL",
    "架构": "TECHNICAL",
    "部署": "TECHNICAL",
    "算法": "TECHNICAL",
}

# ── CJK character detection ──
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CHAR_OR_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]{2,}")


class Compressor:
    """Compress Engram objects into compact context strings.

    Produces strings like:
    ``✅ [user] 模型偏好: 偏爱使用DeepSeek V4 Flash (DECISION+CORE)``
    ``🔒 [reference] NAS路径: NAS挂载于/Volumes/NAS/ (TECHNICAL)``
    """

    def compress(self, engram: Engram) -> str:
        """Compress a single engram into a compact context string.

        Args:
            engram: The engram to compress.

        Returns:
            Compact string suitable for context injection (~30-80 chars).
        """
        parts: list[str] = []

        # 1. Commitment icon
        icon = _COMMITMENT_ICONS.get(engram.commitment, "")
        if icon:
            parts.append(icon)

        # 2. Domain tag
        parts.append(f"[{engram.domain}]")

        # 3. Topic label (first tag or first 3 words)
        label = self._extract_label(engram)
        if label:
            parts.append(f"{label}:")

        # 4. Compressed statement
        compressed_stmt = self._compress_statement(engram.statement, max_chars=60)
        parts.append(compressed_stmt)

        # 5. Flags (extract signals from statement + tags)
        flags = self._detect_flags(engram)
        if flags:
            parts.append(f"({' + '.join(flags[:2])})")

        return " ".join(parts)

    def compress_batch(self, engrams: list[Engram], max_tokens: int = 500) -> str:
        """Compress multiple engrams into a context block.

        Sorts by commitment priority (locked > decided > leaning > exploring),
        then compresses each.  Stops when estimated token count exceeds
        ``max_tokens``.

        Args:
            engrams: List of engrams to compress.
            max_tokens: Maximum estimated token budget.

        Returns:
            Block of compressed strings, one per line.
        """
        priority_order = {"locked": 0, "decided": 1, "leaning": 2, "exploring": 3}
        sorted_engrams = sorted(engrams, key=lambda e: priority_order.get(e.commitment, 99))

        lines: list[str] = []
        est_tokens = 0
        for eng in sorted_engrams:
            line = self.compress(eng)
            line_tokens = self._estimate_tokens(line)
            if est_tokens + line_tokens > max_tokens:
                break
            lines.append(line)
            est_tokens += line_tokens

        return "\n".join(lines)

    # ── Private helpers ──

    def _extract_label(self, engram: Engram) -> str:
        """Extract a topic label from tags or statement prefix."""
        if engram.tags:
            return engram.tags[0]

        # Fallback: first meaningful word from statement
        tokens = _CHAR_OR_TOKEN_RE.findall(engram.statement)
        if tokens:
            return tokens[0][:12]
        return ""

    def _compress_statement(self, statement: str, max_chars: int = 60) -> str:
        """Compress a statement by removing filler words.

        Strategies:
          - Keep key patterns: "X是Y", "X用Y", "X位于Y", "X偏爱Y"
          - Drop particles: 的, 了, 着, 过, 吗, 吧, 呢
          - Drop stop words: the, a, an, is, are, was, were
        """
        # For short statements, return as-is
        if len(statement) <= max_chars:
            return statement

        # Drop common filler particles (Chinese)
        cleaned = statement
        for particle in ["的", "了", "着", "过", "吗", "吧", "呢", "啊", "哦", "就是", "一个"]:
            cleaned = cleaned.replace(particle, "")

        # Compact multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) <= max_chars:
            return cleaned

        return cleaned[: max_chars - 3] + "..."

    def _detect_flags(self, engram: Engram) -> list[str]:
        """Detect MemPalace-style flags from engram statement + metadata."""
        flags: list[str] = []
        text_lower = engram.statement.lower()

        # Scan statement for flag signals
        for signal, flag in _FLAG_SIGNALS.items():
            if signal in text_lower:
                if flag not in flags:
                    flags.append(flag)

        # Commitment-based flags
        if engram.commitment == "locked":
            if "CORE" not in flags:
                flags.insert(0, "CORE")

        # Fact-based flag
        if engram.fact_key:
            if "TECHNICAL" not in flags:
                flags.append("TECHNICAL")

        return flags[:4]  # max 4 flags

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Fast token estimator: CJK chars ≈ 1.5 tokens, ASCII ≈ 1/3 token."""
        cjk_count = len(_CJK_RE.findall(text))
        ascii_count = len(re.findall(r"[a-zA-Z0-9]", text))
        return max(1, int(cjk_count * 1.5 + ascii_count / 3))
