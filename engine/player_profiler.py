"""
Player Profiler — Extracts a unique biomechanical stance signature from
batting pose data. Each player gets a fingerprint based on how they stand,
lift the bat, and set up. Used to recognise returning players across sessions.

Features extracted (all normalised / camera-invariant):
  1. stance_width       — ankle spread as fraction of frame width
  2. hip_shoulder_ratio  — relative width of shoulders vs hips
  3. head_forward        — how far head sits ahead of front hip
  4. grip_height         — wrist height at setup (normalised)
  5. back_lift_height    — peak wrist height during backlift
  6. stance_knee_angle   — front knee bend at setup
  7. face_ratio          — nose-to-ear aspect ratio (profile disambiguation)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Landmarks that matter for stance analysis
STANCE_LANDMARKS = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


def _lmk(landmarks: Dict, name: str) -> Optional[Dict]:
    """Safely get a landmark by name, return None if missing or low visibility."""
    l = landmarks.get(name)
    if l is None or l.get("visibility", 0) < 0.3:
        return None
    return l


def _x(lmk: Optional[Dict]) -> Optional[float]:
    return lmk["x"] if lmk else None


def _y(lmk: Optional[Dict]) -> Optional[float]:
    return lmk["y"] if lmk else None


def _median_valid(values: List[Optional[float]]) -> Optional[float]:
    """Median of non-None values."""
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def _compute_knee_angle(hip, knee, ankle) -> Optional[float]:
    """Compute knee interior angle from 3 landmarks. 180 = straight leg."""
    if not all([hip, knee, ankle]):
        return None
    # Vectors
    a = np.array([hip["x"], hip["y"]])
    b = np.array([knee["x"], knee["y"]])
    c = np.array([ankle["x"], ankle["y"]])
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag < 1e-6:
        return None
    angle = np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))
    return angle


def extract_stance_signature(
    frame_data: List[Dict],
    phase_labels: List[Tuple[int, str]],
    batting_hand: str = "right",
) -> Dict[str, Any]:
    """
    Extract a player's stance signature from a completed analysis session.

    Args:
        frame_data: list of dicts with 'frame', 'landmarks' — from PhaseDetector
        phase_labels: list of (video_frame_idx, phase_name) — from PhaseDetector
        batting_hand: 'right' or 'left'

    Returns:
        dict with stance features and confidence scores
    """
    if batting_hand == "right":
        front, back = "RIGHT", "LEFT"
    else:
        front, back = "LEFT", "RIGHT"

    # Build a set of frames that belong to each phase
    # phase_labels are (frame_idx_in_video, phase_name) but frame_data entries
    # have monotonically increasing 'frame' fields starting from 0.
    # We need to map between them.
    
    # Build a per-entry phase label (aligned with frame_data order)
    phase_by_data_idx = {}  # index into frame_data → phase name
    label_idx = 0
    for i, fd in enumerate(frame_data):
        fidx = fd["frame"]
        # Advance phase label pointer to match or pass this frame
        while label_idx < len(phase_labels) and phase_labels[label_idx][0] < fidx:
            label_idx += 1
        if label_idx < len(phase_labels) and phase_labels[label_idx][0] == fidx:
            phase_by_data_idx[i] = phase_labels[label_idx][1]

    # ── Collect stance phase frames ──
    stance_indices = [
        i for i, phase in phase_by_data_idx.items() if phase == "stance"
    ]
    # Fallback: if no stance phase detected, use first 15 frames
    if not stance_indices and frame_data:
        stance_indices = list(range(min(15, len(frame_data))))

    # ── Collect backlift phase frames ──
    backlift_indices = [
        i for i, phase in phase_by_data_idx.items() if phase == "backlift"
    ]

    # ── Feature 1: Stance width (ankle spread) ──
    ankle_x_vals = []
    for i in stance_indices:
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        a_f = _lmk(lm, f"{front}_ANKLE") or _lmk(lm, f"{front}_FOOT_INDEX") or _lmk(lm, f"{front}_HEEL")
        a_b = _lmk(lm, f"{back}_ANKLE") or _lmk(lm, f"{back}_FOOT_INDEX") or _lmk(lm, f"{back}_HEEL")
        if a_f and a_b:
            ankle_x_vals.append(abs(a_f["x"] - a_b["x"]))
    stance_width = _median_valid(ankle_x_vals) or 0.0

    # ── Feature 2: Hip-shoulder ratio ──
    ratio_vals = []
    for i in stance_indices:
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        ls = _lmk(lm, "LEFT_SHOULDER")
        rs = _lmk(lm, "RIGHT_SHOULDER")
        lh = _lmk(lm, "LEFT_HIP")
        rh = _lmk(lm, "RIGHT_HIP")
        if all([ls, rs, lh, rh]):
            s_dist = abs(ls["x"] - rs["x"])
            h_dist = abs(lh["x"] - rh["x"])
            if h_dist > 0.001:
                ratio_vals.append(s_dist / h_dist)
    hip_shoulder_ratio = _median_valid(ratio_vals) or 0.0

    # ── Feature 3: Head forward (nose ahead of front hip) ──
    head_fwd_vals = []
    for i in stance_indices:
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        nose = _lmk(lm, "NOSE")
        hf = _lmk(lm, f"{front}_HIP")
        hb = _lmk(lm, f"{back}_HIP")
        if nose and hf and hb:
            hip_width = abs(hf["x"] - hb["x"])
            if hip_width > 0.001:
                head_fwd_vals.append((nose["x"] - hf["x"]) / hip_width)
    head_forward = _median_valid(head_fwd_vals) or 0.0

    # ── Feature 4: Grip height (wrist y at setup) ──
    grip_vals = []
    for i in stance_indices:
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        wf = _lmk(lm, f"{front}_WRIST")
        wb = _lmk(lm, f"{back}_WRIST")
        # Use the higher wrist (usually the top hand)
        if wf and wb:
            grip_vals.append(min(wf["y"], wb["y"]))
        elif wf:
            grip_vals.append(wf["y"])
    grip_height = _median_valid(grip_vals) or 0.5
    # Convert so that 0 = low grip, 1 = high grip (easier to compare)
    grip_height_norm = 1.0 - grip_height

    # ── Feature 5: Back lift height ──
    lift_vals = []
    if backlift_indices:
        for i in backlift_indices:
            if i >= len(frame_data):
                continue
            lm = frame_data[i].get("landmarks", {})
            wf = _lmk(lm, f"{front}_WRIST")
            wb = _lmk(lm, f"{back}_WRIST")
            # Highest wrist during backlift (y=0 is top of frame)
            if wf and wb:
                lift_vals.append(min(wf["y"], wb["y"]))
            elif wf:
                lift_vals.append(wf["y"])
    elif frame_data:
        # Fallback: use the minimum wrist y across all frames
        for fd in frame_data:
            lm = fd.get("landmarks", {})
            wf = _lmk(lm, f"{front}_WRIST")
            wb = _lmk(lm, f"{back}_WRIST")
            if wf and wb:
                lift_vals.append(min(wf["y"], wb["y"]))
    min_wrist_y = min(lift_vals) if lift_vals else 0.5
    # 0 = no lift, 1 = maximum lift (wrist at top of frame)
    back_lift_height = 1.0 - min_wrist_y

    # ── Feature 6: Stance knee angle ──
    knee_vals = []
    for i in stance_indices:
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        hip = _lmk(lm, f"{front}_HIP")
        knee = _lmk(lm, f"{front}_KNEE")
        ankle = _lmk(lm, f"{front}_ANKLE")
        angle = _compute_knee_angle(hip, knee, ankle)
        if angle is not None:
            knee_vals.append(angle)
    stance_knee_angle = _median_valid(knee_vals) or 170.0

    # ── Feature 7: Face ratio (profile disambiguation) ──
    face_vals = []
    for i in stance_indices[:10]:  # first few stance frames
        if i >= len(frame_data):
            continue
        lm = frame_data[i].get("landmarks", {})
        nose = _lmk(lm, "NOSE")
        # Use the visible ear (side-on view: one ear will be visible)
        rear = _lmk(lm, f"{back}_EAR")  # back ear = further from camera
        fear = _lmk(lm, f"{front}_EAR")  # front ear = closer
        ear = rear or fear
        if nose and ear:
            dx = abs(nose["x"] - ear["x"])
            dy = abs(nose["y"] - ear["y"])
            if dy > 0.001:
                face_vals.append(dx / dy)
            else:
                face_vals.append(dx * 10)  # approximately vertical alignment
    face_ratio = _median_valid(face_vals) or 0.0

    # ── Confidence / quality score ──
    n_stance = len(stance_indices)
    confidence = min(1.0, n_stance / 30) * 0.7 + (0.3 if backlift_indices else 0.0)

    signature = {
        "stance_width": round(stance_width, 4),
        "hip_shoulder_ratio": round(hip_shoulder_ratio, 4),
        "head_forward": round(head_forward, 4),
        "grip_height": round(grip_height_norm, 4),
        "back_lift_height": round(back_lift_height, 4),
        "stance_knee_angle": round(stance_knee_angle, 1),
        "face_ratio": round(face_ratio, 4),
        "_confidence": round(confidence, 3),
        "_n_stance_frames": n_stance,
        "_n_backlift_frames": len(backlift_indices),
    }
    return signature


def signature_to_vector(signature: Dict) -> np.ndarray:
    """
    Convert a signature dict to a normalised feature vector
    for similarity matching. Omits metadata keys (prefixed with _).

    Each feature is independently normalised to [0, 1] for fair comparison.
    NO L2 normalisation — L2 norm makes all vectors point in similar directions
    and washes out real differences between players when using cosine similarity.
    """
    keys = [
        "stance_width", "hip_shoulder_ratio", "head_forward",
        "grip_height", "back_lift_height", "stance_knee_angle",
        "face_ratio",
    ]
    vals = []
    for k in keys:
        v = signature.get(k, 0.0)
        # Normalise sensible ranges to [0, 1] for fair comparison
        if k == "stance_width":
            v = min(1.0, v / 0.5)  # 0-0.5 → 0-1
        elif k == "hip_shoulder_ratio":
            v = min(1.0, v / 2.0)  # 0.5-2.0 → clamped
        elif k == "head_forward":
            v = max(0.0, min(1.0, (v + 1.0) / 2.0))  # 0..1, properly clamped
        elif k == "grip_height":
            pass  # already 0-1
        elif k == "back_lift_height":
            pass  # already 0-1
        elif k == "stance_knee_angle":
            v = 1.0 - (v - 90) / 90  # 90-180 → 1-0 (bent = 1, straight = 0)
            v = max(0, min(1, v))
        elif k == "face_ratio":
            v = min(1.0, v / 2.0)  # 0-2 → 0-1
        vals.append(v)
    return np.array(vals, dtype=np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two normalised feature vectors."""
    dot = float(np.dot(vec1, vec2))
    n1 = float(np.linalg.norm(vec1))
    n2 = float(np.linalg.norm(vec2))
    if n1 * n2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))  # clamp to [0, 1]


# Population priors for Z-score normalisation, estimated from 5 diverse sessions.
# Each value corresponds to the [0,1]-normalised feature in order:
#   stance_width, hip_shoulder_ratio, head_forward, grip_height,
#   back_lift_height, stance_knee_angle, face_ratio
# These ensure robust Z-score matching even with very few registered profiles.
POPULATION_MEANS = np.array([0.211, 0.753, 0.954, 0.565, 0.671, 0.151, 0.473], dtype=np.float32)
POPULATION_STDS  = np.array([0.063, 0.072, 0.075, 0.058, 0.016, 0.071, 0.293], dtype=np.float32)


def compute_zscore_stats(
    profiles: List[Dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute mean and std for each feature across all registered profiles.
    
    Falls back to population priors (estimated from known sessions) when too
    few profiles are available for reliable statistics.
    
    Returns:
        (means, stds) as numpy arrays, or (None, None) in the degenerate case.
    """
    vecs = []
    for p in profiles:
        sig = p.get("stance_signature", {})
        if sig:
            vecs.append(signature_to_vector(sig))
    
    if len(vecs) >= 3:
        # Compute from actual data once we have enough profiles
        mat = np.array(vecs, dtype=np.float32)
        return np.mean(mat, axis=0), np.std(mat, axis=0)
    elif len(vecs) >= 1:
        # Blend population priors with available data
        mat = np.array(vecs, dtype=np.float32)
        data_mean = np.mean(mat, axis=0)
        data_std  = np.std(mat, axis=0)
        # Weight: more weight on data as profiles grow
        w = min(0.5, len(vecs) * 0.15)
        blended_mean = data_mean * w + POPULATION_MEANS * (1 - w)
        blended_std  = data_std * w + POPULATION_STDS * (1 - w)
        return blended_mean, blended_std
    else:
        return POPULATION_MEANS, POPULATION_STDS


def match_against_profiles(
    signature: Dict,
    profiles: List[Dict],
    threshold: float = 0.50,
) -> Tuple[Optional[str], float]:
    """
    Find the best matching player profile for a given stance signature.

    Uses Z-score standardised Euclidean distance with population priors.
    Distance is converted to a similarity score via 1/(1+distance).

    Args:
        signature: stance signature dict from extract_stance_signature()
        profiles: list of player profile dicts from registry
        threshold: minimum similarity to consider a match (0-1, default 0.70)

    Returns:
        (player_id, similarity_score) or (None, best_score)
    """
    if not profiles or signature.get("_confidence", 0) < 0.3:
        return None, 0.0

    query_vec = signature_to_vector(signature)

    # Always use Z-score standardised Euclidean distance.
    # Falls back to population priors when few/no profiles exist.
    means, stds = compute_zscore_stats(profiles)
    if means is not None:
        query_z = (query_vec - means) / (stds + 1e-10)
    else:
        query_z = query_vec  # fallback (should not happen with priors)

    best_id = None
    best_score = 0.0

    for profile in profiles:
        stored_sig = profile.get("stance_signature", {})
        if not stored_sig:
            continue
        stored_vec = signature_to_vector(stored_sig)
        stored_z = (stored_vec - means) / (stds + 1e-10) if means is not None else stored_vec
        dist = float(np.linalg.norm(query_z - stored_z))

        # Convert to similarity: 1.0 = identical, approaches 0 as distance grows
        score = 1.0 / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_id = profile.get("id")

    if best_score >= threshold and best_id:
        return best_id, best_score
    return None, best_score
