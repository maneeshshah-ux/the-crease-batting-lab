"""
Player Registry — Manages player profiles, session history, and similarity
matching across the entire CREASE system.

Data stored:
  - Player ID and optional name/label
  - Stance signature (7 features) for recognition
  - Historical session metrics (for longitudinal feedback)
  - Session IDs linking to analysis JSONs

Storage:
  Single JSON file (player_registry.json) co-located with the sessions/ dir.
  No server, no login, no privacy concerns — fully local.
"""

import os
import json
import copy
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .player_profiler import match_against_profiles


DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions",
    "player_registry.json",
)


def _load_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict:
    """Load the player registry from disk, returning empty if missing/corrupt."""
    if os.path.exists(registry_path):
        try:
            with open(registry_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "players" in data:
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"players": [], "_version": 2, "_updated": None}


def _save_registry(registry: Dict, registry_path: str = DEFAULT_REGISTRY_PATH):
    """Persist the registry to disk."""
    registry["_updated"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _next_player_id(registry: Dict) -> str:
    """Generate the next available player ID."""
    existing = {p["id"] for p in registry.get("players", []) if "id" in p}
    n = 1
    while f"p_{n:03d}" in existing:
        n += 1
    return f"p_{n:03d}"


def find_or_create_player(
    stance_signature: Dict,
    session_id: str,
    session_metrics: Optional[Dict] = None,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    match_threshold: float = 0.50,
) -> Tuple[Dict, bool]:
    """
    Find an existing player by stance signature, or create a new one.

    Args:
        stance_signature: from player_profiler.extract_stance_signature()
        session_id: e.g. "session_01_105456"
        session_metrics: optional dict of key metrics for history
        registry_path: path to registry JSON
        match_threshold: similarity threshold (0-1)

    Returns:
        (player_dict, is_new: bool)
    """
    registry = _load_registry(registry_path)
    players = registry.get("players", [])

    # Match against existing profiles
    matched_id, score = match_against_profiles(
        stance_signature, players, threshold=match_threshold
    )

    if matched_id:
        # Found existing player
        player = next(p for p in players if p["id"] == matched_id)
        # Update last session
        player["last_session"] = datetime.now().isoformat()
        # Add session ID if not already there
        is_new_session = session_id not in player.get("session_ids", [])
        if is_new_session:
            player.setdefault("session_ids", []).append(session_id)
        # Update historical metrics (only for new sessions to avoid duplicates)
        if session_metrics and is_new_session:
            _update_historical_metrics(player, session_metrics, stance_signature)
        _save_registry(registry, registry_path)
        return player, False
    else:
        # Create new player (pass existing players so ID is unique)
        player = _create_player(stance_signature, session_id, session_metrics, existing_players=players)
        registry.setdefault("players", []).append(player)
        _save_registry(registry, registry_path)
        print(f"  [Player Registry] New player created: {player['id']} "
              f"(similarity to closest match: {score:.2f})")
        return player, True


def _create_player(
    stance_signature: Dict,
    session_id: str,
    session_metrics: Optional[Dict] = None,
    existing_players: Optional[List[Dict]] = None,
) -> Dict:
    """Create a new player profile with a unique ID."""
    now = datetime.now().isoformat()
    player = {
        "id": _next_player_id({"players": existing_players or []}),
        "name": None,  # user can set this later
        "label": "Player ?",
        "created_at": now,
        "last_session": now,
        "session_ids": [session_id],
        "stance_signature": {},
        "historical_metrics": {},
    }
    player["label"] = f"Player {player['id'][-3:]}"

    # Store the stance signature (average of all sessions)
    player["stance_signature"] = {
        k: v for k, v in stance_signature.items() if not k.startswith("_")
    }

    if session_metrics:
        player["historical_metrics"] = {
            k: [v] for k, v in session_metrics.items()
        }

    return player


def _update_historical_metrics(
    player: Dict,
    session_metrics: Dict,
    stance_signature: Dict,
):
    """Append a session's metrics to the player's history."""
    hist = player.setdefault("historical_metrics", {})
    for key, value in session_metrics.items():
        if key.startswith("_"):
            continue
        if key not in hist:
            hist[key] = []
        if isinstance(value, (int, float)):
            hist[key].append(value)

    # Update stance signature as rolling average
    n_sessions = len(player.get("session_ids", []))
    if n_sessions > 0:
        old_sig = player.get("stance_signature", {})
        new_sig = {k: v for k, v in stance_signature.items() if not k.startswith("_")}
        blended = {}
        for key in set(list(old_sig.keys()) + list(new_sig.keys())):
            old_val = old_sig.get(key)
            new_val = new_sig.get(key)
            if old_val is not None and new_val is not None:
                # Weighted average: more weight on newer sessions
                blended[key] = round(
                    (old_val * (n_sessions - 1) + new_val) / n_sessions, 4
                )
            elif new_val is not None:
                blended[key] = new_val
            elif old_val is not None:
                blended[key] = old_val
        player["stance_signature"] = blended


def get_player_history(
    player_id: str,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> Optional[Dict]:
    """Get a player's full history dict."""
    registry = _load_registry(registry_path)
    for p in registry.get("players", []):
        if p.get("id") == player_id:
            return p
    return None


def list_players(registry_path: str = DEFAULT_REGISTRY_PATH) -> List[Dict]:
    """List all registered players with summary info."""
    registry = _load_registry(registry_path)
    summary = []
    for p in registry.get("players", []):
        n_sessions = len(p.get("session_ids", []))
        last = p.get("last_session", "?")[:10]
        hist = p.get("historical_metrics", {})
        latest_score = hist.get("session_score", [None])[-1] if hist.get("session_score") else None
        summary.append({
            "id": p["id"],
            "label": p.get("label", "?"),
            "name": p.get("name"),
            "sessions": n_sessions,
            "last_session": last,
            "latest_score": latest_score,
        })
    return summary


def rename_player(player_id: str, name: str, registry_path: str = DEFAULT_REGISTRY_PATH) -> bool:
    """Set a player's display name."""
    registry = _load_registry(registry_path)
    for p in registry.get("players", []):
        if p.get("id") == player_id:
            p["name"] = name
            _save_registry(registry, registry_path)
            return True
    return False


def compute_session_metrics(result: Dict, report_data: Dict) -> Dict:
    """
    Extract a standard set of metrics from analysis + report for historical tracking.
    These are the values stored per-session for longitudinal comparison.
    """
    ss = result.get("session_summary", {})
    shots = result.get("shot_summary", [])
    bat_speed = result.get("bat_speed", {})
    complete = len([s for s in shots if s.get("has_impact")])

    return {
        "session_score": report_data.get("session_score", 0),
        "bat_speed_avg_kmh": bat_speed.get("swing_avg_kmh", 0),
        "bat_speed_peak_kmh": bat_speed.get("peak_kmh", 0),
        "head_stability_score": ss.get("head_stability_score", 0),
        "avg_front_knee_angle": ss.get("avg_front_knee_angle", 0),
        "avg_spine_angle": ss.get("avg_spine_angle", 0),
        "shot_completion_pct": report_data.get("shot_completion", {}).get("completion_pct", 0),
        "num_shots": len(shots),
        "total_frames": result.get("total_frames", 0),
    }
