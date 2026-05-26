"""
ACT-R + Weibull dual-curve retrieval strength decay.

Blends Plur's ACT-R exponential decay (dominant in short-term,
≤30 days) with the 曙光 memory system's Weibull curve (long-term
plateau beyond 30 days).  Access bonuses counteract decay for
frequently-used memories.

From V2 plan §2.6:
  - ACT-R: decay_constant=0.05/day → 10天=0.61, 30天=0.22
  - Weibull: k=0.5, λ=30天 → 快衰减后平坦
  - Blended: max(ACT-R, Weibull×0.8)
  - Access bonus: min(access_count × 0.05, 0.5)

ECC: Pure computations, no side effects. All functions are idempotent.

Usage:
  from hermes_memory.decay import compute_retrieval_strength
  new_strength = compute_retrieval_strength(engram, now_ms=...)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_memory.models import Engram

# ── ACT-R constants (from Plur) ──
ACT_R_DECAY_CONSTANT: float = 0.05  # daily decay rate
ACT_R_DAYS_TO_SECONDS: float = 86400.0  # seconds per day for ms conversion

# ── Weibull constants (from 曙光记忆系统) ──
WEIBULL_K: float = 0.5       # shape parameter (fast initial, slow later)
WEIBULL_LAMBDA_DAYS: float = 30.0  # scale: 30-day half-life reference

# ── Access bonus ──
ACCESS_BONUS_PER_COUNT: float = 0.05  # per-access retrieval boost
ACCESS_BONUS_MAX: float = 0.5         # cap at +0.5

# ── Commitment decay multipliers ──
# Higher commitment → slower decay (locked memories barely decay)
COMMITMENT_DECAY_MULTIPLIER: dict[str, float] = {
    "locked": 0.2,     # 80% slower
    "decided": 0.5,    # 50% slower
    "leaning": 1.0,    # normal
    "exploring": 1.5,  # 50% faster
}

# ── Minimum strength (never goes below this) ──
MIN_STRENGTH: float = 0.01


def compute_retrieval_strength(engram: "Engram", now_ms: int) -> float:
    """Compute dual-curve retrieval strength for an engram.

    Blends ACT-R exponential decay (short-term dominant) with
    Weibull decay (long-term plateau).  Locked memories decay
    80% slower; exploring memories decay 50% faster.

    Args:
        engram: The engram to compute decay for.
        now_ms: Current time in milliseconds since epoch.

    Returns:
        Retrieval strength in [MIN_STRENGTH, 1.0].
    """
    age_ms = max(0, now_ms - engram.created_at)
    age_days = age_ms / (ACT_R_DAYS_TO_SECONDS * 1000)

    # ACT-R exponential decay
    act_r = engram.retrieval_strength * math.exp(-ACT_R_DECAY_CONSTANT * age_days)

    # Weibull decay: S(t) = exp(-(t/λ)^k)
    if age_days <= 0:
        weibull = 1.0
    else:
        weibull = math.exp(-((age_days / WEIBULL_LAMBDA_DAYS) ** WEIBULL_K))

    # Blend: ACT-R dominates short term, Weibull floors long term
    blended = max(act_r, weibull * 0.8)

    # Access bonus: each access adds 0.05, capped at 0.5
    access_bonus = min(engram.access_count * ACCESS_BONUS_PER_COUNT, ACCESS_BONUS_MAX)

    # Commitment modifier: locked decays slower
    commitment_mult = COMMITMENT_DECAY_MULTIPLIER.get(engram.commitment, 1.0)

    # Apply commitment modifier and access bonus
    # Lower multiplier → slower decay → higher effective strength
    # locked=0.2 → ×5.0, decided=0.5 → ×2.0, leaning=1.0 → ×1.0, exploring=1.5 → ×0.67
    effective_mult = 1.0 / max(commitment_mult, 0.2)
    strength = min(blended * effective_mult, 1.0) + access_bonus

    # Clamp
    return max(min(strength, 1.0), MIN_STRENGTH)


def predict_strength_after_days(
    current_strength: float,
    access_count: int,
    commitment: str,
    days: int,
) -> float:
    """Predict retrieval strength after N days without access.

    Useful for "how long until this memory fades?" queries.

    Args:
        current_strength: Current retrieval strength (0.0-1.0).
        access_count: Current access count.
        commitment: Commitment level.
        days: Number of days to simulate.

    Returns:
        Predicted retrieval strength.
    """
    act_r = current_strength * math.exp(-ACT_R_DECAY_CONSTANT * days)
    weibull = math.exp(-((days / WEIBULL_LAMBDA_DAYS) ** WEIBULL_K))
    blended = max(act_r, weibull * 0.8)

    access_bonus = min(access_count * ACCESS_BONUS_PER_COUNT, ACCESS_BONUS_MAX)
    commitment_mult = COMMITMENT_DECAY_MULTIPLIER.get(commitment, 1.0)
    effective_mult = 1.0 / max(commitment_mult, 0.2)
    strength = min(blended * effective_mult, 1.0) + access_bonus
    return max(min(strength, 1.0), MIN_STRENGTH)


def is_stale(engram: "Engram", now_ms: int, threshold: float = 0.05) -> bool:
    """Check if an engram is stale (below retrieval threshold).

    Stale engrams should be pruned during Dream Cycle.

    Args:
        engram: The engram to check.
        now_ms: Current time in ms.
        threshold: Strength below which engram is considered stale.

    Returns:
        True if engram should be pruned.
    """
    strength = compute_retrieval_strength(engram, now_ms)
    return strength < threshold and not engram.pinned
