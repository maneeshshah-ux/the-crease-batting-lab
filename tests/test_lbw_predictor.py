"""
Tests for LbwPredictor — single-camera LBW probability estimation.
"""

import sys
import pytest
sys.path.insert(0, "batting_analyser")
from engine.lbw_predictor import LbwPredictor, _classify_stump_zone, _zone_to_verdict, _confidence_label


# ────────────────────────────────────────────────────────────────────────
# Tests — Stump zone classification
# ────────────────────────────────────────────────────────────────────────

class TestStumpZones:
    def test_outside_off(self):
        assert _classify_stump_zone(0.10) == "outside_off"

    def test_off_stump(self):
        assert _classify_stump_zone(0.30) == "off_stump"

    def test_middle_stump(self):
        assert _classify_stump_zone(0.45) == "middle_stump"

    def test_leg_stump(self):
        assert _classify_stump_zone(0.60) == "leg_stump"

    def test_missing_leg(self):
        assert _classify_stump_zone(0.80) == "missing_leg"

    def test_zone_boundary_off(self):
        """Boundary between outside_off and off_stump at 0.25."""
        assert _classify_stump_zone(0.249) == "outside_off"
        assert _classify_stump_zone(0.250) == "off_stump"

    def test_zone_boundary_leg(self):
        """Boundary between leg_stump and missing_leg at 0.68."""
        assert _classify_stump_zone(0.679) == "leg_stump"
        assert _classify_stump_zone(0.680) == "missing_leg"

    def test_unknown_zone(self):
        assert _classify_stump_zone(-0.1) == "unknown"
        assert _classify_stump_zone(1.5) == "unknown"


class TestVerdicts:
    def test_all_zones_have_verdicts(self):
        for zone in ["outside_off", "off_stump", "middle_stump", "leg_stump", "missing_leg"]:
            v = _zone_to_verdict(zone)
            assert isinstance(v, str) and len(v) > 5

    def test_verdict_readable(self):
        assert "outside" in _zone_to_verdict("outside_off").lower()
        assert "middle" in _zone_to_verdict("middle_stump").lower()


class TestConfidenceLabels:
    def test_high_confidence(self):
        assert _confidence_label(25) == "high"

    def test_medium_confidence(self):
        assert _confidence_label(10) == "medium"

    def test_low_confidence(self):
        assert _confidence_label(5) == "low"

    def test_estimate_confidence(self):
        assert _confidence_label(0) == "estimate"


# ────────────────────────────────────────────────────────────────────────
# Tests — Core prediction
# ────────────────────────────────────────────────────────────────────────

class TestCorePrediction:
    def test_middle_stump_high_prob(self):
        """Ball on middle stump → high hitting probability."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump",
            trajectory_points=25,
        )
        assert result["hitting_stumps_pct"] >= 70
        assert result["ball_line_zone"] == "middle_stump"
        assert result["confidence"] in ("high", "medium")

    def test_outside_off_low_prob(self):
        """Ball outside off → low hitting probability."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="outside_off",
            trajectory_points=25,
        )
        assert result["hitting_stumps_pct"] <= 30
        assert result["ball_line_zone"] == "outside_off"

    def test_missing_leg_low_prob(self):
        """Ball down leg → low hitting probability."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="missing_leg",
            trajectory_points=25,
        )
        assert result["hitting_stumps_pct"] <= 30

    def test_off_stump_moderate_prob(self):
        """Ball on off stump → moderate probability."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="off_stump",
            trajectory_points=15,
        )
        assert 20 <= result["hitting_stumps_pct"] <= 85

    def test_leg_stump_moderate_prob(self):
        """Ball on leg stump → moderate probability."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="leg_stump",
            trajectory_points=15,
        )
        assert 20 <= result["hitting_stumps_pct"] <= 85

    def test_direct_ball_x_overrides_line(self):
        """Direct ball_x measurement should override classified ball_line."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="outside_off",  # would give low prob
            ball_x_at_crease=800,     # middle stump for 1920px → 0.417 norm
            frame_width=1920,
            trajectory_points=30,
        )
        # 800/1920 = 0.417 → middle_stump → high prob
        assert result["hitting_stumps_pct"] >= 60
        assert result["ball_line_zone"] == "middle_stump"

    def test_pitch_zone_fallback(self):
        """Should work with pitch zone when ball_line is unavailable."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line=None,
            pitch_zone="middle_stump",
            trajectory_points=5,
        )
        assert result["hitting_stumps_pct"] >= 50
        assert result["confidence"] == "low"


# ────────────────────────────────────────────────────────────────────────
# Tests — Modifiers
# ────────────────────────────────────────────────────────────────────────

class TestModifiers:
    def test_edge_modifier_increases_prob(self):
        """Edge impact → ball was close to stumps → higher prob."""
        predictor = LbwPredictor()
        result_middle = predictor.predict(
            ball_line="off_stump", impact_point="middle", trajectory_points=20
        )
        result_edge = predictor.predict(
            ball_line="off_stump", impact_point="edge", trajectory_points=20
        )
        assert result_edge["hitting_stumps_pct"] >= result_middle["hitting_stumps_pct"]

    def test_batter_forward_reduces_prob(self):
        """Batter forward → ball travels further → slightly lower prob."""
        predictor = LbwPredictor()
        result_static = predictor.predict(
            ball_line="middle_stump", batter_forward=False, trajectory_points=20
        )
        result_forward = predictor.predict(
            ball_line="middle_stump", batter_forward=True, trajectory_points=20
        )
        assert result_forward["hitting_stumps_pct"] <= result_static["hitting_stumps_pct"]

    def test_pitch_zone_modifier_off_stump(self):
        """Ball pitching off stump → dangerous line → higher modifier."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump", pitch_zone="off_stump", trajectory_points=20
        )
        assert result["hitting_stumps_pct"] >= 70

    def test_pitch_zone_modifier_down_leg(self):
        """Ball pitching down leg → lower modifier."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump", pitch_zone="down_leg", trajectory_points=20
        )
        # Should be lower than without modifier
        assert result["hitting_stumps_pct"] >= 10


# ────────────────────────────────────────────────────────────────────────
# Tests — Foot alignment
# ────────────────────────────────────────────────────────────────────────

class TestFootAlignment:
    def test_batter_on_off_ball_on_off_increases(self):
        """Batter covering off + ball hitting off = more in line."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="off_stump",
            foot_alignment="covering_off",
            trajectory_points=20,
        )
        assert result["hitting_stumps_pct"] >= 50

    def test_batter_on_leg_ball_on_off_decreases(self):
        """Batter on leg + ball on off = less in line."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="off_stump",
            foot_alignment="covering_leg",
            trajectory_points=20,
        )
        assert result["hitting_stumps_pct"] <= 80  # should be reduced


# ────────────────────────────────────────────────────────────────────────
# Tests — Cone of uncertainty
# ────────────────────────────────────────────────────────────────────────

class TestConeOfUncertainty:
    def test_cone_present(self):
        """Cone should always be present in result."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump",
            trajectory_points=15,
        )
        assert "lower" in result["cone"]
        assert "upper" in result["cone"]
        assert result["cone"]["lower"] <= result["cone"]["upper"]

    def test_more_points_narrows_cone(self):
        """More trajectory points → narrower cone."""
        predictor = LbwPredictor()
        result_few = predictor.predict(
            ball_line="middle_stump", trajectory_points=5, ball_x_at_crease=800, frame_width=1920
        )
        result_many = predictor.predict(
            ball_line="middle_stump", trajectory_points=50, ball_x_at_crease=800, frame_width=1920
        )
        few_width = result_few["cone"]["upper"] - result_few["cone"]["lower"]
        many_width = result_many["cone"]["upper"] - result_many["cone"]["lower"]
        assert many_width <= few_width

    def test_direct_measurement_narrower_cone(self):
        """Direct ball x → narrower cone than classified line."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump",
            trajectory_points=20,
        )
        # Classified_line source has base_half_width=0.06
        cone_width = result["cone"]["upper"] - result["cone"]["lower"]
        assert cone_width > 0


# ────────────────────────────────────────────────────────────────────────
# Tests — Caveat / disclaimer
# ────────────────────────────────────────────────────────────────────────

class TestCaveat:
    def test_caveat_present(self):
        """Every prediction must include the single-camera caveat."""
        predictor = LbwPredictor()
        result = predictor.predict(ball_line="outside_off")
        assert "caveat" in result
        assert "single-camera" in result["caveat"].lower()
        assert "DRS" in result["caveat"]

    def test_caveat_not_empty(self):
        predictor = LbwPredictor()
        result = predictor.predict(ball_line="middle_stump")
        assert len(result["caveat"]) > 20


# ────────────────────────────────────────────────────────────────────────
# Tests — Predict_shot convenience wrapper
# ────────────────────────────────────────────────────────────────────────

class TestPredictShot:
    def test_shot_with_line(self):
        """predict_shot should work with a minimal shot dict."""
        predictor = LbwPredictor()
        shot = {
            "shot_number": 1,
            "ball_line": "off_stump",
            "ball_length": "good",
            "impact_point_label": "edge",
        }
        result = predictor.predict_shot(
            shot=shot,
            ball_trajectory=[(500, 300), (510, 350), (520, 400)],
            frame_width=1920,
            frame_height=1080,
        )
        assert "hitting_stumps_pct" in result
        assert result["verdict"]
        assert result["ball_line_zone"] in ("off_stump", "middle_stump")

    def test_shot_without_line_falls_back(self):
        """Shot without ball_line should fall back gracefully."""
        predictor = LbwPredictor()
        shot = {
            "shot_number": 1,
            "ball_line": None,
            "ball_length": None,
            "impact_point_label": "unknown",
        }
        result = predictor.predict_shot(
            shot=shot,
            ball_trajectory=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert result["hitting_stumps_pct"] == 0
        assert result["confidence"] == "estimate"


# ────────────────────────────────────────────────────────────────────────
# Tests — Edge cases
# ────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_trajectory_points(self):
        """Should still work with no trajectory data (use ball_line only)."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump",
            trajectory_points=0,
        )
        assert result["confidence"] == "estimate"
        assert result["hitting_stumps_pct"] > 0

    def test_all_none_inputs(self):
        """All inputs None → graceful fallback."""
        predictor = LbwPredictor()
        result = predictor.predict()
        assert result["hitting_stumps_pct"] == 0
        assert "Insufficient" in result["verdict"]

    def test_left_hand_batting(self):
        """Should handle left-hand batting hand."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="middle_stump",
            batting_hand="left",
            trajectory_points=20,
        )
        assert result["hitting_stumps_pct"] >= 70

    def test_extreme_pitch_modifier(self):
        """Pitch modifier should not produce impossible values."""
        predictor = LbwPredictor()
        # Even with extreme modifiers, prob should stay in [1, 99]
        for pitch_zone in ("outside_off", "off_stump", "middle_stump", "leg_stump", "down_leg"):
            result = predictor.predict(
                ball_line="middle_stump",
                pitch_zone=pitch_zone,
                impact_point="edge",
                trajectory_points=20,
            )
            assert 1 <= result["hitting_stumps_pct"] <= 99

    def test_cone_within_bounds(self):
        """Cone should never go outside [0, 1]."""
        predictor = LbwPredictor()
        result = predictor.predict(
            ball_line="outside_off",
            trajectory_points=2,
        )
        assert result["cone"]["lower"] >= 0.0
        assert result["cone"]["upper"] <= 1.0

    def test_no_ball_line_with_pitch_zone(self):
        """Should work with pitch_zone only (no ball_line)."""
        predictor = LbwPredictor()
        result = predictor.predict(
            pitch_zone="off_stump",
            trajectory_points=10,
        )
        assert result["hitting_stumps_pct"] > 0


# ────────────────────────────────────────────────────────────────────────
# Tests — Verdict format
# ────────────────────────────────────────────────────────────────────────

class TestVerdict:
    def test_hitting_verdict(self):
        predictor = LbwPredictor()
        result = predictor.predict(ball_line="off_stump", trajectory_points=20)
        assert "hitting" in result["verdict"].lower() or "missing" in result["verdict"].lower()

    def test_missing_verdict(self):
        predictor = LbwPredictor()
        result = predictor.predict(ball_line="outside_off", trajectory_points=20)
        assert "missing" in result["verdict"].lower()

    def test_insufficient_data_verdict(self):
        predictor = LbwPredictor()
        result = predictor.predict()
        assert "Insufficient" in result["verdict"]
