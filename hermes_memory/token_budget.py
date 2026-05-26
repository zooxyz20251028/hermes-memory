"""
Token-budget-controlled engram injection from GBrain's budget executor.

Selects engrams for context injection with commitment-weighted ordering
under a hard token cap.  Follows V2 plan §2.5 phase 3:
  1. Sort by commitment priority (pinned > locked > decided > leaning > exploring)
  2. Within same commitment: sort by retrieval_strength desc
  3. Fill until token budget exhausted
  4. Hard cap: never exceed max_tokens

ECC: Pure computation, no side effects.

Usage:
  from hermes_memory.token_budget import select_for_injection
  selected = select_for_injection(engrams, max_tokens=2000)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_memory.models import Engram

# ── Commitment priority order (lower = higher priority) ──
_COMMITMENT_PRIORITY: dict[str, int] = {
    "locked": 0,
    "decided": 1,
    "leaning": 2,
    "exploring": 3,
}

# ── Pinned boost ──
PINNED_PRIORITY: int = -1  # pinned gets priority -1 (above locked)

# ── Token estimation ──
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CHARS_PER_TOKEN_CJK: float = 1.5
_CHARS_PER_TOKEN_ASCII: float = 3.0

# ── Default budgets ──
DEFAULT_CONSERVATIVE: int = 1000   # fast injection
DEFAULT_BALANCED: int = 2000       # standard
DEFAULT_COMPREHENSIVE: int = 4000  # deep recall


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    CJK characters ≈ 1.5 tokens each, ASCII ≈ 1 token per 3 chars.
    """
    cjk_count = len(_CJK_RE.findall(text))
    ascii_count = len(re.findall(r"[a-zA-Z0-9]", text))
    return max(1, int(cjk_count * _CHARS_PER_TOKEN_CJK + ascii_count / _CHARS_PER_TOKEN_ASCII))


def select_for_injection(
    engrams: list["Engram"],
    max_tokens: int = DEFAULT_BALANCED,
    *,
    include_pinned: bool = True,
) -> list["Engram"]:
    """Select engrams for context injection under token budget.

    Priority ordering:
      1. Pinned engrams (always included if include_pinned=True)
      2. Commitment: locked > decided > leaning > exploring
      3. Retrieval strength (descending) within same commitment

    Args:
        engrams: Candidate engrams from multi-path search.
        max_tokens: Maximum estimated token budget.
        include_pinned: Always include pinned engrams first.

    Returns:
        Selected engrams in injection order.
    """
    if not engrams:
        return []

    # Separate pinned
    pinned: list["Engram"] = []
    rest: list["Engram"] = []
    for e in engrams:
        if e.pinned and include_pinned:
            pinned.append(e)
        else:
            rest.append(e)

    # Sort rest by commitment priority, then retrieval_strength desc
    rest.sort(
        key=lambda e: (
            _COMMITMENT_PRIORITY.get(e.commitment, 99),
            -e.retrieval_strength,
        ),
    )

    selected: list["Engram"] = []
    total_est_tokens = 0

    # Pinned first
    for e in pinned:
        est = estimate_tokens(e.statement)
        if total_est_tokens + est <= max_tokens:
            selected.append(e)
            total_est_tokens += est
        else:
            break  # can't fit more pinned — stop

    # Then commitment-ordered rest
    for e in rest:
        est = estimate_tokens(e.statement)
        if total_est_tokens + est > max_tokens:
            continue  # skip this one, try next (may be shorter)
        selected.append(e)
        total_est_tokens += est

    return selected


def budget_stats(engrams: list["Engram"], max_tokens: int) -> dict:
    """Return injection budget statistics.

    Args:
        engrams: All candidate engrams.
        max_tokens: Token budget limit.

    Returns:
        Dict with 'selected', 'rejected', 'total_tokens', 'budget', 'fill_pct'.
    """
    selected = select_for_injection(engrams, max_tokens)
    total_tokens = sum(estimate_tokens(e.statement) for e in selected)
    return {
        "selected": len(selected),
        "rejected": len(engrams) - len(selected),
        "total_tokens": total_tokens,
        "budget": max_tokens,
        "fill_pct": round(total_tokens / max_tokens * 100, 1) if max_tokens > 0 else 0,
    }
