"""
Crease Currency — Token-based usage gating for the CREASE Batting Lab.

Every action that consumes compute costs one or more tokens (Crease Credits).
Tokens are tied to subscription tier and reset monthly.

Token allocation by tier:
  free       →   3 analyses / month  (analyses_limit=3 in profiles table)
  pro        →  unlimited
  enterprise →  unlimited + priority queue

Token consumption per action:
  COST_ANALYSIS     = 1   (one full video analysis)
  COST_PDF_REPORT   = 0   (free — it's generated from cached data)
  COST_HIGHLIGHTS   = 0   (free — same session data)

The system reads/writes `analyses_used` and `analyses_limit` in the Supabase
`profiles` table.  When Supabase is not configured, all checks pass (dev mode).

Usage:
    from crease_currency import can_analyse, deduct_analysis_token, get_balance

    if not can_analyse(user_id):
        return error("You've used all your free analyses this month.")
    result = analyser.analyse_video(...)
    deduct_analysis_token(user_id)
"""

from __future__ import annotations

import logging
from typing import Optional

from supabase_client import get_supabase, is_configured

log = logging.getLogger(__name__)

# Token costs per action
COST_ANALYSIS   = 1
COST_PDF_REPORT = 0
COST_HIGHLIGHTS = 0

# Tier → monthly limit (None = unlimited)
TIER_LIMITS: dict[str, Optional[int]] = {
    "free":       3,
    "pro":        None,   # unlimited
    "enterprise": None,   # unlimited + priority
}

# Default for new/unknown users
DEFAULT_LIMIT = 3


def get_balance(user_id: str) -> dict:
    """
    Return the current token balance for a user.

    Returns:
        {
            "tier":       "free" | "pro" | "enterprise",
            "used":       int,
            "limit":      int | None,   # None = unlimited
            "remaining":  int | None,   # None = unlimited
            "can_analyse": bool,
        }
    """
    if not user_id or not is_configured():
        # No auth / dev mode — always allow
        return {
            "tier": "dev",
            "used": 0,
            "limit": None,
            "remaining": None,
            "can_analyse": True,
        }

    sb = get_supabase()
    if sb is None:
        return _offline_balance()

    try:
        resp = (
            sb.table("profiles")
            .select("subscription_tier, analyses_used, analyses_limit")
            .eq("id", user_id)
            .single()
            .execute()
        )
        profile = resp.data
    except Exception as exc:
        log.warning("[CreaseCurrency] Could not read profile for %s: %s", user_id, exc)
        return _offline_balance()

    if not profile:
        return _offline_balance()

    tier  = profile.get("subscription_tier", "free")
    used  = profile.get("analyses_used", 0) or 0
    limit = profile.get("analyses_limit", DEFAULT_LIMIT)

    # Pro/enterprise: override limit to None (unlimited)
    if tier in ("pro", "enterprise"):
        limit = None

    remaining = None if limit is None else max(0, limit - used)
    can = (limit is None) or (used < limit)

    return {
        "tier":        tier,
        "used":        used,
        "limit":       limit,
        "remaining":   remaining,
        "can_analyse": can,
    }


def can_analyse(user_id: str) -> bool:
    """Quick check — True if the user has tokens remaining."""
    return get_balance(user_id)["can_analyse"]


def deduct_analysis_token(user_id: str) -> bool:
    """
    Increment analyses_used by COST_ANALYSIS (1).

    Returns True on success, False if Supabase write fails.
    Always returns True in dev/offline mode (no-op).
    """
    if not user_id or not is_configured():
        return True  # dev mode passthrough

    if COST_ANALYSIS == 0:
        return True

    sb = get_supabase()
    if sb is None:
        return True

    try:
        # Atomic increment using RPC if available, otherwise read-modify-write
        balance = get_balance(user_id)
        new_used = balance["used"] + COST_ANALYSIS
        sb.table("profiles").update({
            "analyses_used": new_used,
        }).eq("id", user_id).execute()
        log.info(
            "[CreaseCurrency] Deducted %d token(s) for user %s. New total: %d/%s",
            COST_ANALYSIS, user_id, new_used,
            str(balance["limit"]) if balance["limit"] is not None else "∞",
        )
        return True

    except Exception as exc:
        log.error("[CreaseCurrency] Failed to deduct token for user %s: %s", user_id, exc)
        return False


def reset_monthly_usage(user_id: str) -> bool:
    """
    Reset analyses_used to 0.  Call this from your monthly Stripe webhook
    (invoice.paid) or a scheduled task to refresh the token pool.
    """
    if not is_configured():
        return True

    sb = get_supabase()
    if sb is None:
        return False

    try:
        sb.table("profiles").update({"analyses_used": 0}).eq("id", user_id).execute()
        log.info("[CreaseCurrency] Monthly usage reset for user %s", user_id)
        return True
    except Exception as exc:
        log.error("[CreaseCurrency] Reset failed for user %s: %s", user_id, exc)
        return False


def set_tier_limit(user_id: str, tier: str) -> bool:
    """Update a user's tier and analyses_limit when they upgrade/downgrade."""
    if not is_configured():
        return True

    sb = get_supabase()
    if sb is None:
        return False

    limit = TIER_LIMITS.get(tier, DEFAULT_LIMIT)
    # Unlimited tiers still need a high number in the DB (null is tricky with RLS)
    db_limit = 99999 if limit is None else limit

    try:
        sb.table("profiles").update({
            "subscription_tier": tier,
            "analyses_limit": db_limit,
        }).eq("id", user_id).execute()
        log.info("[CreaseCurrency] Set tier=%s limit=%s for user %s", tier, db_limit, user_id)
        return True
    except Exception as exc:
        log.error("[CreaseCurrency] set_tier_limit failed for %s: %s", user_id, exc)
        return False


# ── Private helpers ────────────────────────────────────────────────────────────

def _offline_balance() -> dict:
    """Returned when Supabase is unreachable — allow everything."""
    return {
        "tier":        "offline",
        "used":        0,
        "limit":       None,
        "remaining":   None,
        "can_analyse": True,
    }
