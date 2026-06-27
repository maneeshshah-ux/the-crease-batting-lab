"""
Person Tracker — Sliding-window person classifier for cricket net videography.

Core problem: In a front-on (non-striker end) cricket video, multiple people
appear in frame:
  - BATTER (primary subject): far end of pitch, small-ish, stationary
  - BOWLER: runs in from behind camera, varies in size, bowling action
  - WICKETKEEPER / FIELDER: at edges, minimal movement

Since MediaPipe Pose returns only the SINGLE most prominent person per frame,
we classify THAT person using smart heuristics. A sliding window with temporal
smoothing prevents label flickering.

Feature extraction per frame:
  - Foot/nose position (y)         → height in frame
  - Bounding box height            → how much frame they occupy
  - Hand separation                → together (bat grip) vs apart
  - Arm/wrist height               → raised arm (bowling action)
  - Movement vector (dx, dy)       → stationary vs running

Temporal rules:
  - Labels must persist min_stability_frames before changing
  - Bowler follow-through persists for ~15 frames after delivery
  - Uncertain frames use mode of last 10 frames
"""

from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class PersonTracker:
    """
    Frame-by-frame person classifier with temporal smoothing.

    Labels:
        "batter"                 — Primary batting subject
        "bowler_approach"        — Bowler running in
        "bowler_delivery"        — Bowler at delivery stride
        "bowler_follow_through"  — Just after delivery
        "wk_fielder"             — Wicketkeeper or close fielder
        "empty"                  — No person detected
        "uncertain"              — Features are ambiguous
    """

    # ── Default feature thresholds (tuned for front-on cricket) ──
    _BATTER_FOOT_Y_MIN = 0.42       # Batter feet in lower half of frame
    _BATTER_HAND_DIST_MAX = 0.10    # Hands together on bat grip
    _BATTER_WRIST_Y_MIN = 0.32      # Wrists not raised high
    _BATTER_MOVEMENT_MAX = 0.03     # Minimal inter-frame movement

    _BOWLER_ARM_RAISED_RATIO = 0.85  # Wrist above shoulder ratio
    _BOWLER_WRIST_Y_MAX = 0.28       # Wrist high in frame
    _BOWLER_DOWNWARD_VEL = 0.015     # Fast downward movement
    _BOWLER_HAND_DIST_MIN = 0.12     # Hands apart (no bat)

    _WK_BBOX_HEIGHT_MAX = 0.20       # Small in frame
    _WK_FOOT_Y_MAX = 0.40            # Upper portion of frame

    def __init__(
        self,
        window_size: int = 30,
        batting_hand: str = "right",
        camera_view: str = "front_on",
        min_stability_frames: int = 5,
    ):
        """
        Args:
            window_size: Frames kept for temporal smoothing.
            batting_hand: "right" or "left".
            camera_view: "side_off", "side_leg", "front_on", "behind", "angled".
            min_stability_frames: Minimum frames before label can change.
        """
        self.window_size = window_size
        self.batting_hand = batting_hand
        self.camera_view = camera_view
        self.min_stability_frames = min_stability_frames

        # Sliding window
        self.frame_history = deque(maxlen=window_size)
        self.person_bbox_history = deque(maxlen=15)

        # Current stable state
        self.current_label = "empty"
        self.label_since_frame = -1  # -1 means "not yet set" (allows first classification)
        self.total_frames_processed = 0

        # Previous frame tracking
        self.prev_landmarks: Optional[Dict] = None

        # Front / back assignment for batting hand
        if batting_hand == "right":
            self.front = "LEFT"
            self.back = "RIGHT"
        else:
            self.front = "RIGHT"
            self.back = "LEFT"

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        pose_result: Dict[str, Any],
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Classify the person detected in a single pose-estimation result.

        Args:
            pose_result: Output from PoseEstimator.process_frame().
            frame_shape: (height, width) of the raw frame.

        Returns:
            Enhanced dict with additional keys:
                person_label: One of the class labels above.
                person_confidence: Float in [0, 1].
                is_batter: Boolean shortcut (True if label == "batter").
            Original pose_result keys are preserved.
        """
        h, w = frame_shape
        self.total_frames_processed += 1
        frame_idx = self.total_frames_processed - 1

        landmarks = pose_result.get("landmarks", {})
        result = {
            **pose_result,
            "person_label": "empty",
            "person_confidence": 0.0,
            "is_batter": False,
        }

        # Determine raw label:
        #   - No person detected → "empty"
        #   - Person detected → classify using features
        if not pose_result.get("success", False) or not landmarks:
            raw_label = "empty"
            confidence = 1.0
        else:
            features = self._extract_features(landmarks, h, w)
            raw_label, confidence = self._classify(features, landmarks, h, w)

        # ALWAYS apply temporal smoothing — even empty frames pass through
        # so rules like "bowler follow-through persists through empty frames" work.
        smoothed_label = self._smooth_label(raw_label, confidence)

        # Backfill result
        result["person_label"] = smoothed_label
        result["person_confidence"] = round(confidence, 3)
        result["is_batter"] = (smoothed_label == "batter")

        # Update original pose_result for backward-compatible code
        pose_result["is_batter"] = result["is_batter"]
        pose_result["person_label"] = smoothed_label
        pose_result["person_confidence"] = result["person_confidence"]

        if landmarks:
            self.prev_landmarks = landmarks
        return result

    def get_current_label(self) -> str:
        """Most recent stable label."""
        return self.current_label

    def get_label_history(self, n: int = 30) -> List[str]:
        """Last *n* labels (most recent last)."""
        recent = list(self.frame_history)
        return [r["label"] for r in recent[-n:]] if recent else []

    def is_batter_active(self) -> bool:
        """Is the batter currently the detected person?"""
        recent = self.get_label_history(10)
        return bool(recent and recent[-1] == "batter")

    def is_bowler_active(self) -> bool:
        """Is any bowler variant currently in frame?"""
        recent = self.get_label_history(5)
        return bool(recent and any(l.startswith("bowler") for l in recent))

    def reset(self):
        """Clear all internal state."""
        self.frame_history.clear()
        self.person_bbox_history.clear()
        self.current_label = "empty"
        self.label_since_frame = -1
        self.total_frames_processed = 0
        self.prev_landmarks = None

    # ────────────────────────────────────────────────────────────────
    # Feature extraction
    # ────────────────────────────────────────────────────────────────

    def _extract_features(self, landmarks: Dict, frame_h: int,
                          frame_w: int) -> Dict[str, float]:
        """Build a feature vector from the MediaPipe landmark dict."""
        f: Dict[str, float] = {}

        # -- Foot y position -----------------------------------------
        foot_y_vals = []
        for key in ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
                     "LEFT_ANKLE", "RIGHT_ANKLE"):
            lm = landmarks.get(key, {})
            if lm.get("visibility", 0) > 0.3:
                foot_y_vals.append(lm["y"])
        f["foot_y_mean"] = float(np.mean(foot_y_vals)) if foot_y_vals else 0.5
        f["has_feet"] = float(len(foot_y_vals) > 0)

        # -- Nose position -------------------------------------------
        nose = landmarks.get("NOSE", {})
        f["nose_y"] = nose.get("y", 0.5)
        f["nose_x"] = nose.get("x", 0.5)

        # -- Bounding box (from all visible landmarks) ----------------
        all_x = [v["x"] for v in landmarks.values()
                 if v.get("visibility", 0) > 0.3]
        all_y = [v["y"] for v in landmarks.values()
                 if v.get("visibility", 0) > 0.3]
        if all_x and all_y:
            f["bbox_height"] = max(all_y) - min(all_y)
            f["bbox_width"] = max(all_x) - min(all_x)
            f["bbox_center_x"] = (max(all_x) + min(all_x)) / 2.0
            f["bbox_center_y"] = (max(all_y) + min(all_y)) / 2.0
        else:
            f.update(bbox_height=0.3, bbox_width=0.1,
                     bbox_center_x=0.5, bbox_center_y=0.5)

        # -- Hand separation -----------------------------------------
        lw = landmarks.get("LEFT_WRIST", {})
        rw = landmarks.get("RIGHT_WRIST", {})
        if lw.get("x") is not None and rw.get("x") is not None:
            dx = lw["x"] - rw["x"]
            dy = lw["y"] - rw["y"]
            f["hand_distance"] = float(np.sqrt(dx**2 + dy**2))
        else:
            f["hand_distance"] = 0.5

        # -- Arm / wrist height ---------------------------------------
        wrist_y_vals = [
            v["y"] for k in ("LEFT_WRIST", "RIGHT_WRIST")
            if (v := landmarks.get(k, {})).get("y") is not None
        ]
        elbow_y_vals = [
            v["y"] for k in ("LEFT_ELBOW", "RIGHT_ELBOW")
            if (v := landmarks.get(k, {})).get("y") is not None
        ]
        shoulder_y_vals = [
            v["y"] for k in ("LEFT_SHOULDER", "RIGHT_SHOULDER")
            if (v := landmarks.get(k, {})).get("y") is not None
        ]
        f["min_wrist_y"] = min(wrist_y_vals) if wrist_y_vals else 0.5
        f["min_elbow_y"] = min(elbow_y_vals) if elbow_y_vals else 0.5

        # Wrist-above-shoulder ratio (< 1 = raised arm)
        min_shoulder_y = min(shoulder_y_vals) if shoulder_y_vals else 1.0
        if min_shoulder_y > 0 and f["min_wrist_y"] < min_shoulder_y:
            f["wrist_shoulder_ratio"] = f["min_wrist_y"] / min_shoulder_y
        else:
            f["wrist_shoulder_ratio"] = 1.0  # not raised

        # -- Inter-frame movement ------------------------------------
        if self.prev_landmarks is not None:
            pn = self.prev_landmarks.get("NOSE", {})
            cn = landmarks.get("NOSE", {})
            if pn.get("x") is not None and cn.get("x") is not None:
                dx = cn["x"] - pn["x"]
                dy = cn["y"] - pn["y"]
                f["movement"] = float(np.sqrt(dx**2 + dy**2))
                f["movement_dx"] = float(dx)
                f["movement_dy"] = float(dy)
            else:
                f.update(movement=0.0, movement_dx=0.0, movement_dy=0.0)
        else:
            f.update(movement=0.0, movement_dx=0.0, movement_dy=0.0)

        return f

    # ────────────────────────────────────────────────────────────────
    # Per-frame classification
    # ────────────────────────────────────────────────────────────────

    def _classify(self, f: Dict[str, float], landmarks: Dict,
                  h: int, w: int) -> Tuple[str, float]:
        """
        Raw single-frame classification.

        Returns:
            (label, confidence) — confidence is an unbounded score that
            is clamped to [0, 1] later.
        """
        # ── Batter score ────────────────────────────────────────────
        batter_score = 0.0

        # 1. Foot position: batter feet are in lower half of frame
        foot_y = f["foot_y_mean"]
        if foot_y > self._BATTER_FOOT_Y_MIN:       # > 0.42 → strong signal
            batter_score += 0.30
        elif foot_y > 0.35:                        # 0.35–0.42 → weak signal
            batter_score += 0.12
        else:
            batter_score -= 0.20                   # < 0.35 → NOT batter

        # 2. Hand proximity: batter has hands together on bat handle
        hand_dist = f["hand_distance"]
        if hand_dist < self._BATTER_HAND_DIST_MAX:     # < 0.10 → together
            batter_score += 0.25
        elif hand_dist < self._BATTER_HAND_DIST_MAX + 0.05:  # 0.10–0.15
            batter_score += 0.12
        # Hands very far apart → NOT batter (penalty applied elsewhere)

        # 3. Wrist height: batter wrists not raised above shoulder
        if f["min_wrist_y"] > self._BATTER_WRIST_Y_MIN:       # > 0.32
            batter_score += 0.15
        elif f["min_wrist_y"] > self._BATTER_WRIST_Y_MIN - 0.08:  # 0.24–0.32
            batter_score += 0.06
        else:
            batter_score -= 0.10                    # wrists very high → NOT batter

        # 4. Movement: batter is relatively still (except during shot)
        if f["movement"] < self._BATTER_MOVEMENT_MAX:
            batter_score += 0.15
        elif f["movement"] < self._BATTER_MOVEMENT_MAX * 2:
            batter_score += 0.06

        # 5. Size check (view-aware): batter occupies consistent portion of frame
        bbox_h = f["bbox_height"]
        if self.camera_view in ("front_on", "behind"):
            if 0.20 < bbox_h < 0.50:
                batter_score += 0.20
            elif bbox_h < 0.15:          # too small → not batter
                batter_score -= 0.15

        # ── Bowler score ────────────────────────────────────────────
        bowler_score = 0.0
        wsr = f.get("wrist_shoulder_ratio", 1.0)
        if wsr < self._BOWLER_ARM_RAISED_RATIO:
            bowler_score += 0.30
        elif wsr < 0.95:
            bowler_score += 0.12

        if f["min_wrist_y"] < self._BOWLER_WRIST_Y_MAX:
            bowler_score += 0.20
        elif f["min_wrist_y"] < 0.35:
            bowler_score += 0.08

        if f.get("movement_dy", 0) > self._BOWLER_DOWNWARD_VEL:
            bowler_score += 0.25
        elif f.get("movement_dy", 0) > 0.008:
            bowler_score += 0.10

        if f["hand_distance"] > self._BOWLER_HAND_DIST_MIN:
            bowler_score += 0.15
        elif f["hand_distance"] > self._BOWLER_HAND_DIST_MIN - 0.04:
            bowler_score += 0.06

        # Bbox variance (bowler size changes as they run)
        prev_bboxes = list(self.person_bbox_history)
        if len(prev_bboxes) >= 5:
            var = float(np.var([b.get("bbox_height", 0.3) for b in prev_bboxes]))
            if var > 0.004:
                bowler_score += 0.15

        # ── WK / fielder score ─────────────────────────────────────
        wk_score = 0.0
        if bbox_h < self._WK_BBOX_HEIGHT_MAX and f["foot_y_mean"] < self._WK_FOOT_Y_MAX:
            wk_score = 0.6
        elif bbox_h < self._WK_BBOX_HEIGHT_MAX + 0.08:
            wk_score = 0.3

        # ── Decision ────────────────────────────────────────────────
        scores = [
            ("batter", batter_score),
            ("bowler_approach", bowler_score),
            ("wk_fielder", wk_score),
        ]
        # Only consider uncertain if all scores are low
        best_label, best_score = max(scores, key=lambda x: x[1])

        if best_score < 0.3:
            return "uncertain", best_score

        # Refine bowler sub-type
        if best_label == "bowler_approach":
            if wsr < 0.70 or f.get("movement_dy", 0) > 0.03:
                best_label = "bowler_delivery"

        confidence = min(1.0, best_score)
        return best_label, confidence

    # ────────────────────────────────────────────────────────────────
    # Temporal smoothing
    # ────────────────────────────────────────────────────────────────

    def _smooth_label(self, raw_label: str, confidence: float) -> str:
        """
        Apply temporal rules to prevent flickering.

        Rules:
          1. "uncertain" → mode of last 10 non-uncertain frames.
          2. Labels must persist min_stability_frames before changing.
          3. Bowler delivery → follow-through transition.
        """
        frame_idx = self.total_frames_processed - 1
        self._update_history(raw_label, confidence, frame_idx)

        recent = list(self.frame_history)
        if not recent:
            return raw_label

        # Rule 1: replace uncertain with mode of recent labels
        if raw_label == "uncertain":
            recent_labels = [r["label"] for r in recent[-10:]
                             if r["label"] != "uncertain"]
            if recent_labels:
                return Counter(recent_labels).most_common(1)[0][0]
            return raw_label

        # Rule 2 (HIGH PRIORITY): bowler follow-through persistence
        # Checked BEFORE the stability gate so that "empty" or "uncertain"
        # frames during a bowler delivery are correctly labelled as
        # follow-through rather than being blocked by the stability gate.
        if (self.current_label in ("bowler_delivery", "bowler_follow_through")
                and raw_label in ("bowler_approach", "uncertain", "empty")):
            frames_since = frame_idx - self.label_since_frame
            if frames_since < 15:
                return "bowler_follow_through"

        # Rule 3: stability gate — don't flip too fast
        # First classification always passes (label_since_frame == -1)
        if self.current_label != raw_label and self.label_since_frame >= 0:
            frames_since = frame_idx - self.label_since_frame
            if frames_since < self.min_stability_frames:
                return self.current_label

        self.current_label = raw_label
        self.label_since_frame = frame_idx
        return raw_label

    # ────────────────────────────────────────────────────────────────
    # History management
    # ────────────────────────────────────────────────────────────────

    def _update_history(self, label: str, confidence: float, frame_idx: int):
        self.frame_history.append({
            "frame": frame_idx,
            "label": label,
            "confidence": confidence,
        })
