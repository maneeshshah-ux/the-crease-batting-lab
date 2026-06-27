"""
Tests for the front-on shot classifier — including modern/new-age shots.
"""

import sys
import pytest
from batting_analyser.engine.shot_classifier import ShotClassifier, ShotType


# ── Test fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sc():
    return ShotClassifier(batting_hand="right", frame_height=1080)


BASE_FEATURES = {
    "foot_movement": "forward",
    "swing_path_angle": 30.0,
    "bat_face": "straight",
    "ball_line": "middle_stump",
    "ball_length": "full",
    "front_knee_min": 160.0,
    "bat_speed_max": 70.0,
}


# ── Traditional shots ─────────────────────────────────────────────────────

class TestTraditionalShots:
    def test_cover_drive(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=20.0, bat_face="open",
                 ball_line="outside_off")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.COVER_DRIVE
        assert r[1] >= 0.7

    def test_on_drive_middle(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=15.0, bat_face="closed",
                 ball_line="middle_stump")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.ON_DRIVE
        assert r[1] >= 0.7

    def test_straight_drive(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=10.0, bat_face="straight",
                 ball_line="middle_stump")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.STRAIGHT_DRIVE
        assert r[1] >= 0.7

    def test_square_cut(self, sc):
        f = dict(BASE_FEATURES, foot_movement="back", swing_path_angle=70.0,
                 bat_face="open", ball_line="outside_off", ball_length="short")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.SQUARE_CUT
        assert r[1] >= 0.7

    def test_pull(self, sc):
        f = dict(BASE_FEATURES, foot_movement="back", swing_path_angle=75.0,
                 bat_face="closed", ball_line="leg_stump", ball_length="short")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.PULL
        assert r[1] >= 0.7

    def test_defensive_block(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=2.0, foot_movement="forward")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.DEFENSIVE_BLOCK

    def test_leave(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=None, ball_line="outside_off",
                 foot_movement="no_stride")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.LEAVE

    def test_sweep(self, sc):
        f = dict(BASE_FEATURES, front_knee_min=88.0, swing_path_angle=60.0,
                 bat_face="closed", ball_line="leg_stump", bat_speed_max=60.0)
        r = sc._decision_tree(f)
        assert r[0] == ShotType.SWEEP
        assert r[1] >= 0.7

    def test_glance(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=45.0, bat_face="closed",
                 ball_line="leg_stump")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.GLANCE

    def test_flick(self, sc):
        f = dict(BASE_FEATURES, swing_path_angle=40.0, bat_face="open",
                 ball_line="off_stump", foot_movement="forward")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.FLICK


# ── Modern / new-age shots ────────────────────────────────────────────────

class TestModernShots:
    def test_reverse_sweep(self, sc):
        """Knee down + open bat face = reverse sweep."""
        f = dict(BASE_FEATURES, front_knee_min=85.0, bat_face="open",
                 swing_path_angle=50.0, ball_line="middle_stump",
                 bat_speed_max=60.0)
        r = sc._decision_tree(f)
        assert r[0] == ShotType.REVERSE_SWEEP
        assert r[1] >= 0.7

    def test_slog_sweep(self, sc):
        """Knee down + high bat speed = slog sweep."""
        f = dict(BASE_FEATURES, front_knee_min=90.0, bat_speed_max=150.0,
                 swing_path_angle=60.0, bat_face="closed",
                 ball_line="leg_stump")
        r = sc._decision_tree(f)
        assert r[0] == ShotType.SLOG_SWEEP
        assert r[1] >= 0.7

    def test_lap_shot_straight_face(self, sc):
        """Knee down + neutral (straight) bat face = lap shot."""
        f = dict(BASE_FEATURES, front_knee_min=92.0, bat_face="straight",
                 swing_path_angle=50.0, ball_line="middle_stump",
                 bat_speed_max=60.0)
        r = sc._decision_tree(f)
        assert r[0] == ShotType.LAP_SHOT
        assert r[1] >= 0.7

    def test_lap_shot_down_leg(self, sc):
        """Knee down + ball down leg = lap shot."""
        f = dict(BASE_FEATURES, front_knee_min=90.0, bat_face="closed",
                 swing_path_angle=55.0, ball_line="down_leg",
                 bat_speed_max=60.0)
        r = sc._decision_tree(f)
        assert r[0] == ShotType.LAP_SHOT
        assert r[1] >= 0.7

    def test_upper_cut(self, sc):
        """Back foot + angle 35-60° + ball outside off = upper cut."""
        f = dict(BASE_FEATURES, foot_movement="back", swing_path_angle=48.0,
                 bat_face="open", ball_line="outside_off",
                 ball_length="short", front_knee_min=160.0)
        r = sc._decision_tree(f)
        assert r[0] == ShotType.UPPER_CUT
        assert r[1] >= 0.7

    def test_upper_cut_not_front_foot(self, sc):
        """Upper cut should NOT trigger on front foot (→ flick instead)."""
        f = dict(BASE_FEATURES, foot_movement="forward", swing_path_angle=48.0,
                 bat_face="open", ball_line="outside_off",
                 front_knee_min=160.0)
        r = sc._decision_tree(f)
        assert r[0] != ShotType.UPPER_CUT

    def test_upper_cut_not_low_angle(self, sc):
        """Upper cut should NOT trigger for low (<35°) angle."""
        f = dict(BASE_FEATURES, foot_movement="back", swing_path_angle=30.0,
                 bat_face="open", ball_line="outside_off",
                 ball_length="short", front_knee_min=160.0)
        r = sc._decision_tree(f)
        assert r[0] != ShotType.UPPER_CUT

    def test_upper_cut_not_high_angle(self, sc):
        """Upper cut should NOT trigger for full horizontal (>60°) angle."""
        f = dict(BASE_FEATURES, foot_movement="back", swing_path_angle=65.0,
                 bat_face="open", ball_line="outside_off",
                 ball_length="short", front_knee_min=160.0)
        r = sc._decision_tree(f)
        assert r[0] != ShotType.UPPER_CUT

    def test_sweep_not_confused_with_slog(self, sc):
        """Regular-speed sweep should NOT be slog_sweep."""
        f = dict(BASE_FEATURES, front_knee_min=88.0, bat_speed_max=60.0,
                 swing_path_angle=60.0, bat_face="closed",
                 ball_line="leg_stump")
        r = sc._decision_tree(f)
        assert r[0] != ShotType.SLOG_SWEEP
        assert r[0] == ShotType.SWEEP

    def test_sweep_not_confused_with_reverse(self, sc):
        """Closed-face sweep should NOT be reverse_sweep."""
        f = dict(BASE_FEATURES, front_knee_min=88.0, bat_face="closed",
                 swing_path_angle=60.0, ball_line="leg_stump",
                 bat_speed_max=60.0)
        r = sc._decision_tree(f)
        assert r[0] != ShotType.REVERSE_SWEEP
        assert r[0] == ShotType.SWEEP


# ── Utility tests ─────────────────────────────────────────────────────────

class TestUtilities:
    def test_all_types_have_icons(self, sc):
        for t in ShotType:
            icon = sc.shot_type_icon(t.value)
            assert icon is not None and len(icon) > 0

    def test_all_types_have_descriptions(self, sc):
        for t in ShotType:
            desc = sc.shot_type_description(t.value)
            assert desc is not None and len(desc) > 10

    def test_unknown_icon_fallback(self, sc):
        assert sc.shot_type_icon("nonexistent") == "❓"

    def test_unknown_description_fallback(self, sc):
        assert "Unidentified" in sc.shot_type_description("nonexistent")

    def test_new_types_in_enum(self):
        values = {e.value for e in ShotType}
        for expected in ("reverse_sweep", "slog_sweep", "lap_shot",
                         "ramp", "upper_cut"):
            assert expected in values, f"Missing {expected} in ShotType enum"

    def test_left_hand_batting_hand(self):
        """Left-hander: off side is viewer right, leg side is viewer left."""
        sc_left = ShotClassifier(batting_hand="left", frame_height=1080)
        assert sc_left._off_side == "right"
