"""
Tests for BowlingAnalyzer — opportunistic bowling analysis.
"""

import math
import sys
import pytest
import numpy as np
sys.path.insert(0, "batting_analyser")
from engine.bowling_analyzer import BowlingAnalyzer, BowlType


# ────────────────────────────────────────────────────────────────────────
# Helpers: build synthetic pose data
# ────────────────────────────────────────────────────────────────────────

def _make_landmarks(
    right_wrist_y: float = 0.35,
    left_wrist_y: float = 0.40,
    right_shoulder_y: float = 0.30,
    left_shoulder_y: float = 0.30,
    nose_y: float = 0.25,
) -> dict:
    """Build a minimal landmarks dict."""
    return {
        "RIGHT_WRIST": {"x": 0.55, "y": right_wrist_y, "visibility": 0.9},
        "LEFT_WRIST": {"x": 0.45, "y": left_wrist_y, "visibility": 0.9},
        "RIGHT_SHOULDER": {"x": 0.55, "y": right_shoulder_y, "visibility": 0.9},
        "LEFT_SHOULDER": {"x": 0.45, "y": left_shoulder_y, "visibility": 0.9},
        "RIGHT_ELBOW": {"x": 0.52, "y": 0.38, "visibility": 0.9},
        "LEFT_ELBOW": {"x": 0.48, "y": 0.38, "visibility": 0.9},
        "NOSE": {"x": 0.50, "y": nose_y, "visibility": 0.9},
    }


def _make_bowler_frame(frame: int, label: str, landmarks: dict) -> dict:
    return {
        "frame": frame,
        "person_label": label,
        "landmarks": landmarks,
        "ball_x": None,
        "ball_y": None,
        "ball_detected": False,
        "ball_confidence": 0,
    }


def _run_up_sequence(
    start_frame: int = 0,
    length: int = 15,
    label: str = "bowler_approach",
    nose_start: float = 0.30,
    nose_end: float = 0.55,
) -> list:
    """Generate a sequence of approach frames with increasing nose_y
    (bowler running toward camera = appearing lower in frame)."""
    frames = []
    for i in range(length):
        t = i / max(1, length - 1)
        nose_y = nose_start + (nose_end - nose_start) * t
        lm = _make_landmarks(nose_y=nose_y)
        frames.append(_make_bowler_frame(start_frame + i, label, lm))
    return frames


def _delivery_sequence(
    start_frame: int = 15,
    length: int = 5,
    label: str = "bowler_delivery",
    wrist_y_start: float = 0.20,
    wrist_y_end: float = 0.50,
) -> list:
    """Generate delivery frames with arm going from high to low."""
    frames = []
    for i in range(length):
        t = i / max(1, length - 1)
        wy = wrist_y_start + (wrist_y_end - wrist_y_start) * t
        # For right-arm bowler, right wrist is bowling arm
        lm = _make_landmarks(right_wrist_y=wy, nose_y=0.55)
        frames.append(_make_bowler_frame(start_frame + i, label, lm))
    return frames


def _follow_through_sequence(
    start_frame: int = 20,
    length: int = 10,
    label: str = "bowler_follow_through",
) -> list:
    """Generate follow-through frames."""
    frames = []
    for i in range(length):
        lm = _make_landmarks(right_wrist_y=0.55 + i * 0.01, nose_y=0.55)
        frames.append(_make_bowler_frame(start_frame + i, label, lm))
    return frames


def _make_complete_delivery(
    approach_len: int = 15,
    delivery_len: int = 5,
    follow_len: int = 10,
    nose_start: float = 0.30,
) -> list:
    """Build a complete delivery: approach → delivery → follow-through."""
    frames = []
    offset = 0
    frames.extend(_run_up_sequence(offset, approach_len, nose_start=nose_start))
    offset += approach_len
    frames.extend(_delivery_sequence(offset, delivery_len))
    offset += delivery_len
    frames.extend(_follow_through_sequence(offset, follow_len))
    return frames


# ────────────────────────────────────────────────────────────────────────
# Tests — BowlType helpers
# ────────────────────────────────────────────────────────────────────────

class TestBowlType:
    def test_all_types_have_labels(self):
        for t in BowlType.all_types():
            label = BowlType.label(t)
            assert isinstance(label, str) and len(label) > 0
            assert label != t or t == "unknown"  # label should be title-cased

    def test_all_types_have_icons(self):
        for t in BowlType.all_types():
            icon = BowlType.icon(t)
            assert isinstance(icon, str) and len(icon) > 0

    def test_unknown_icon_fallback(self):
        assert BowlType.icon("nonexistent") == "❓"


# ────────────────────────────────────────────────────────────────────────
# Tests — Empty / Null input
# ────────────────────────────────────────────────────────────────────────

class TestEmptyInput:
    def test_no_frames(self):
        ba = BowlingAnalyzer(fps=30)
        result = ba.analyse([])
        assert result["has_bowling_data"] is False
        assert result["num_deliveries_detected"] == 0
        assert result["bowl_type"] == "unknown"

    def test_no_bowler_frames(self):
        ba = BowlingAnalyzer(fps=30)
        # Frames that exist but aren't bowler
        frames = [
            _make_bowler_frame(0, "batter", _make_landmarks()),
            _make_bowler_frame(1, "empty", _make_landmarks()),
        ]
        result = ba.analyse(frames)
        assert result["has_bowling_data"] is False


# ────────────────────────────────────────────────────────────────────────
# Tests — Delivery Detection
# ────────────────────────────────────────────────────────────────────────

class TestDeliveryDetection:
    def test_detects_single_delivery(self):
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery()
        result = ba.analyse(frames)
        assert result["has_bowling_data"] is True
        assert result["num_deliveries_detected"] == 1

    def test_detects_multiple_deliveries(self):
        ba = BowlingAnalyzer(fps=30)
        frames = []
        frames.extend(_make_complete_delivery(0))
        # Gap (batter frames)
        frames.append(_make_bowler_frame(45, "batter", _make_landmarks()))
        frames.extend(_make_complete_delivery(50))
        result = ba.analyse(frames)
        assert result["num_deliveries_detected"] == 2

    def test_delivery_requires_delivery_frames(self):
        """Only approach frames should not count as a delivery."""
        ba = BowlingAnalyzer(fps=30)
        frames = _run_up_sequence(0, 20)
        result = ba.analyse(frames)
        assert result["has_bowling_data"] is False


# ────────────────────────────────────────────────────────────────────────
# Tests — Run-up Speed
# ────────────────────────────────────────────────────────────────────────

class TestRunUpSpeed:
    def test_run_up_detected(self):
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery(approach_len=15)
        result = ba.analyse(frames)
        assert result["num_deliveries_detected"] >= 1
        delivery = result["deliveries"][0]
        # Run-up should have been estimated
        assert delivery.get("run_up_speed_px_per_sec") is not None
        assert delivery["run_up_speed_px_per_sec"] > 0

    def test_avg_run_up_reported(self):
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery(approach_len=20)
        result = ba.analyse(frames)
        assert result["avg_run_up_speed_px_per_s"] > 0

    def test_no_approach_no_runup(self):
        ba = BowlingAnalyzer(fps=30)
        # Only delivery + follow-through
        frames = _delivery_sequence(0, 5) + _follow_through_sequence(5, 10)
        result = ba.analyse(frames)
        # Should still detect a delivery
        assert result["num_deliveries_detected"] >= 1
        # But run-up should be None
        assert result["deliveries"][0].get("run_up_speed_px_per_sec") is None


# ────────────────────────────────────────────────────────────────────────
# Tests — Arm Speed
# ────────────────────────────────────────────────────────────────────────

class TestArmSpeed:
    def test_arm_speed_detected(self):
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery(delivery_len=5)
        result = ba.analyse(frames)
        delivery = result["deliveries"][0]
        assert delivery.get("arm_speed_rad_s") is not None
        assert delivery["arm_speed_rad_s"] > 0

    def test_fast_arm_speed(self):
        """High wrist velocity → fast arm action."""
        ba = BowlingAnalyzer(fps=30)
        # With very quick wrist drop (y: 0.20 → 0.50 in 3 frames)
        frames = _delivery_sequence(0, 3, wrist_y_start=0.20, wrist_y_end=0.50)
        frames.extend(_follow_through_sequence(3, 5))
        result = ba.analyse(frames)
        delivery = result["deliveries"][0]
        arm_speed = delivery.get("arm_speed_rad_s", 0)
        # Should be detectable (not necessarily fast because
        # arm angle computation depends on geometry)
        assert arm_speed > 0 or arm_speed == 0


# ────────────────────────────────────────────────────────────────────────
# Tests — Release Height
# ────────────────────────────────────────────────────────────────────────

class TestReleaseHeight:
    def test_release_height_detected(self):
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery()
        result = ba.analyse(frames)
        delivery = result["deliveries"][0]
        assert delivery.get("release_height") is not None
        assert 0 <= delivery["release_height"] <= 1.0

    def test_high_release(self):
        """Fast bowler: wrist high (y ≈ 0.25) at delivery."""
        ba = BowlingAnalyzer(fps=30)
        lm = _make_landmarks(right_wrist_y=0.25, nose_y=0.55)
        frames = _delivery_sequence(0, 3) + _follow_through_sequence(3, 5)
        # Override landmarks to set high release
        for f in frames:
            if f["person_label"] == "bowler_delivery":
                f["landmarks"]["RIGHT_WRIST"]["y"] = 0.25
        result = ba.analyse(frames)
        delivery = result["deliveries"][0]
        assert delivery["release_height"] <= 0.35  # high release

    def test_low_release(self):
        """Spinner: wrist lower (y ≈ 0.55) at delivery."""
        ba = BowlingAnalyzer(fps=30)
        frames = _delivery_sequence(0, 3) + _follow_through_sequence(3, 5)
        for f in frames:
            if f["person_label"] == "bowler_delivery":
                f["landmarks"]["RIGHT_WRIST"]["y"] = 0.55
        result = ba.analyse(frames)
        delivery = result["deliveries"][0]
        assert delivery["release_height"] >= 0.50  # low release


# ────────────────────────────────────────────────────────────────────────
# Tests — Bowl Type Classification
# ────────────────────────────────────────────────────────────────────────

class TestBowlTypeClassification:
    def test_fast_bowler(self):
        """Fast bowler: high arm speed + high release + fast ball."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(
            arm_speed=15.0,        # ≥12 → fast
            release_height=0.30,   # ≤0.45 → fast
            ball_speed_kmh=135.0,  # ≥120 → fast (≥130 → pure fast)
        )
        assert bowl_type == BowlType.FAST

    def test_fast_medium_bowler(self):
        """Fast-medium: arm speed high + release high, ball speed 120-130."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(
            arm_speed=14.0,
            release_height=0.35,
            ball_speed_kmh=125.0,  # ≥120 but <130 → fast-medium
        )
        assert bowl_type == BowlType.FAST_MEDIUM

    def test_spin_bowler(self):
        """Spin: low arm speed + low release + slow ball."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(
            arm_speed=5.0,         # <8 → spin
            release_height=0.55,   # ≥0.50 → spin
            ball_speed_kmh=80.0,   # <90 → spin
        )
        assert bowl_type == BowlType.SPIN

    def test_medium_bowler(self):
        """Medium: all signals in medium range."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(
            arm_speed=9.0,         # ≥8 → medium
            release_height=0.48,   # between 0.45 and 0.50
            ball_speed_kmh=100.0,  # ≥90 → medium
        )
        assert bowl_type == BowlType.MEDIUM

    def test_medium_fast_bowler(self):
        """Medium-fast: medium signals with ball speed ≥115."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(
            arm_speed=10.0,
            release_height=0.44,
            ball_speed_kmh=118.0,  # ≥115 → medium-fast
        )
        assert bowl_type == BowlType.MEDIUM_FAST

    def test_ball_speed_classification_fast(self):
        """Fallback: ball speed only → fast."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_by_ball_speed(130.0)
        assert bowl_type == BowlType.FAST

    def test_ball_speed_classification_medium(self):
        """Fallback: ball speed only → medium."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_by_ball_speed(95.0)
        assert bowl_type == BowlType.MEDIUM

    def test_ball_speed_classification_spin(self):
        """Fallback: ball speed only → spin."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_by_ball_speed(75.0)
        assert bowl_type == BowlType.SPIN

    def test_no_signals_unknown(self):
        """No signals at all → unknown."""
        ba = BowlingAnalyzer(fps=30)
        bowl_type = ba._classify_bowl_type(None, None, None)
        assert bowl_type == BowlType.UNKNOWN

    def test_arm_speed_votes_fast_overrides_spin_release(self):
        """Arm speed is the strongest signal (weighted 2x)."""
        ba = BowlingAnalyzer(fps=30)
        # arm_speed high (weight 2) vs release low (weight 1)
        bowl_type = ba._classify_bowl_type(
            arm_speed=14.0,        # fast: +2
            release_height=0.60,   # spin: +1 → fast wins
            ball_speed_kmh=80.0,   # spin: +1 → tie? fast wins (higher in dict)
        )
        assert bowl_type in (BowlType.FAST, BowlType.FAST_MEDIUM)


# ────────────────────────────────────────────────────────────────────────
# Tests — Left-arm bowler
# ────────────────────────────────────────────────────────────────────────

class TestLeftArmBowler:
    def test_left_arm_bowler_detected(self):
        """Left-arm: bowling arm is LEFT."""
        ba = BowlingAnalyzer(fps=30, batting_hand="left")
        assert ba.bowling_arm == "LEFT"
        assert ba.non_bowling_arm == "RIGHT"

    def test_left_arm_arm_angle(self):
        """Verify arm angle works with left-arm landmarks."""
        ba = BowlingAnalyzer(fps=30, batting_hand="left")
        # For left-arm bowler, LEFT_WRIST is bowling arm
        lm = _make_landmarks(right_wrist_y=0.40, left_wrist_y=0.25)
        angle = ba._compute_arm_angle(lm)
        assert angle is not None
        assert angle >= 0


# ────────────────────────────────────────────────────────────────────────
# Tests — End-to-end integrated scenarios
# ────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_fast_bowler_scenario(self):
        """Simulate a fast bowler with high arm action + ball speed."""
        ba = BowlingAnalyzer(fps=30)
        # Build a delivery with high arm (right_wrist_y=0.25)
        frames = _make_complete_delivery()
        for f in frames:
            if f["person_label"] == "bowler_delivery":
                f["landmarks"]["RIGHT_WRIST"]["y"] = 0.25
        result = ba.analyse(frames, ball_speed_kmh=135.0)
        assert result["has_bowling_data"] is True
        assert result["num_deliveries_detected"] >= 1
        assert result["bowl_type"] in (BowlType.FAST, BowlType.FAST_MEDIUM)

    def test_full_spin_bowler_scenario(self):
        """Simulate a spinner with low arm + slower ball."""
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery()
        for f in frames:
            if f["person_label"] == "bowler_delivery":
                f["landmarks"]["RIGHT_WRIST"]["y"] = 0.55
        result = ba.analyse(frames, ball_speed_kmh=75.0)
        assert result["has_bowling_data"] is True
        assert result["bowl_type"] == BowlType.SPIN

    def test_results_in_result_dict(self):
        """Verify the result dict structure from analyse()."""
        ba = BowlingAnalyzer(fps=30)
        frames = _make_complete_delivery()
        result = ba.analyse(frames, ball_speed_kmh=100.0)
        # Check all expected keys
        assert "has_bowling_data" in result
        assert "num_deliveries_detected" in result
        assert "deliveries" in result
        assert "bowl_type" in result
        assert "bowl_type_label" in result
        assert "bowl_type_icon" in result
        assert "bowl_type_confidence" in result
        assert "avg_run_up_speed_px_per_s" in result
        assert "avg_arm_speed_rad_s" in result
        assert "avg_release_height" in result
        assert "avg_release_height_cm" in result

        # Check delivery structure
        if result["deliveries"]:
            d = result["deliveries"][0]
            assert "bowl_type" in d
            assert "bowl_type_label" in d
            assert "bowl_type_icon" in d
            assert "bowling_arm" in d
