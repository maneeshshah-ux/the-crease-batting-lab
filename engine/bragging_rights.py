"""
Bragging Rights Engine — comparative stats that make every session
feel like an achievement worth sharing.

Produces short, punchy one-liners like:
  "Your top speed beats 73% of batters this week"
  "You played 5 different shot types — more variety than 68% of sessions"
  "Best shot: Cover Drive (92% confidence)"

These are incorporated into:
  - Scorecard images
  - Social share text
  - Session detail hero area
"""

from __future__ import annotations

from typing import Any, Dict, List


# "Exciting" shot types that are more share-worthy
EXCITING_SHOTS = {
    "slog_sweep", "reverse_sweep", "ramp", "upper_cut",
    "lap_shot", "cover_drive", "straight_drive", "pull",
}


def compute_bragging_rights(
    shot_summary: List[Dict[str, Any]],
    session_summary: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute comparative / bragging stats from session data.

    Args:
        shot_summary: List of shot dicts from the analysis.
        session_summary: Aggregate metrics dict.
        extra: Dict with keys like ``{"bat_speed": {...}}``.

    Returns:
        Bragging rights dict::

            {
                "best_shot_type": "cover_drive",
                "best_shot_confidence": 0.92,
                "best_shot_label": "Cover Drive",
                "excitement_score": 75,        # 0-100
                "diversity_score": 68,          # 0-100
                "power_score": 82,              # 0-100
                "total_exciting_shots": 3,
                "unique_shot_types": 5,
                "one_liners": [
                    "Your cover drive is your best shot (92% confidence)",
                    "You played 5 different shot types — great variety!",
                    "3 exciting shots in this session",
                ],
            }
    """
    if not shot_summary:
        return {
            "best_shot_type": None,
            "best_shot_confidence": 0,
            "best_shot_label": None,
            "excitement_score": 0,
            "diversity_score": 0,
            "power_score": 0,
            "total_exciting_shots": 0,
            "unique_shot_types": 0,
            "one_liners": ["Upload a session to see bragging rights!"],
        }

    # ── Best shot ───────────────────────────────────────────────────
    best_shot = max(
        shot_summary,
        key=lambda s: s.get("classification_confidence", 0) or 0,
        default=None,
    )
    best_type = best_shot.get("shot_type", "unknown") if best_shot else "unknown"
    best_conf = (best_shot.get("classification_confidence", 0) or 0) if best_shot else 0
    best_label = best_type.replace("_", " ").title()

    # ── Shot type diversity ─────────────────────────────────────────
    unique_types = {
        s.get("shot_type") for s in shot_summary
        if s.get("shot_type") and s["shot_type"] != "unknown"
    }
    n_unique = len(unique_types)
    diversity_score = min(n_unique * 15, 100)

    # ── Excitement count ────────────────────────────────────────────
    exciting_count = sum(
        1 for s in shot_summary
        if s.get("shot_type") in EXCITING_SHOTS
    )
    excitement_score = min(exciting_count * 20, 100)

    # ── Power score (bat speed relative) ────────────────────────────
    bat_speed = extra.get("bat_speed", {})
    if isinstance(bat_speed, dict):
        top_speed = bat_speed.get("swing_max_kmh", 0) or bat_speed.get("swing_avg_kmh", 0) or 0
    else:
        top_speed = bat_speed or 0

    # Rough scale: 60 km/h = 40%, 80 km/h = 60%, 100 km/h = 80%, 120+ = 100%
    if top_speed > 0:
        power_score = min(int((top_speed / 120) * 100), 100)
    else:
        power_score = 0

    # ── One-liners ──────────────────────────────────────────────────
    one_liners = []

    if best_type and best_conf > 0.5:
        one_liners.append(
            f"Your {best_label} is your best shot ({best_conf * 100:.0f}% confidence)"
        )

    if n_unique >= 5:
        one_liners.append(f"You played {n_unique} different shot types — great variety!")
    elif n_unique >= 3:
        one_liners.append(f"{n_unique} shot types used. Try adding the sweep or cut next!")
    elif n_unique <= 2:
        one_liners.append("Try expanding your shot range — variety keeps bowlers guessing")

    if exciting_count >= 3:
        one_liners.append(f"{exciting_count} exciting shots in this session")
    elif exciting_count >= 1:
        one_liners.append("You played some power shots!")

    if top_speed > 100:
        one_liners.append(f"Bat speed of {top_speed:.0f} km/h — serious power!")
    elif top_speed > 80:
        one_liners.append(f"Top bat speed {top_speed:.0f} km/h — keep building that power")
    elif top_speed > 0:
        one_liners.append(f"Top bat speed {top_speed:.0f} km/h — building strength")

    # Cap at 3 for display
    one_liners = one_liners[:3]

    return {
        "best_shot_type": best_type,
        "best_shot_confidence": round(best_conf, 2),
        "best_shot_label": best_label,
        "excitement_score": excitement_score,
        "diversity_score": diversity_score,
        "power_score": power_score,
        "total_exciting_shots": exciting_count,
        "unique_shot_types": n_unique,
        "one_liners": one_liners,
    }
