"""
Batting Analyser Engine — Zero-cost cricket batting analysis.

Key modules:
    analyser          — Main BattingAnalyser pipeline orchestrator
    pose_estimator    — MediaPipe-based pose detection
    person_tracker    — Sliding-window person classifier (batter/bowler/WK)
    ball_tracker      — Ball detection & trajectory tracking
    bat_analyzer      — Bat swing inference from hand landmarks
    phase_detector    — Batting shot phase detection (7 phases)
    metrics           — Biomechanical metrics calculator (view-aware)
    visualizer        — Video overlay renderer (Fox-Sport inspired)
    benchmarks        — Player comparison data & view-aware lookup
    player_profiler   — Stance signature extraction & matching
    player_registry   — Player profile persistence & retrieval
    report_generator  — PDF coaching report generation
    voiceover         — Audio coaching feedback
    longitudinal_feedback — Cross-session progress tracking
"""

from .person_tracker import PersonTracker
from .ball_tracker import FrontOnBallTracker, BallTracker, TrackingPhase
from .shot_classifier import ShotClassifier, ShotType
from .front_on_metrics import (
    compute_front_on_frame_metrics,
    estimate_bat_face,
    estimate_foot_stump_alignment,
    estimate_lateral_trigger,
    estimate_head_line_sync,
    estimate_balance_direction,
    estimate_impact_point,
    estimate_shoulder_alignment,
)
from .bragging_rights import compute_bragging_rights
from .highlight_reel import HighlightReel
from .scorecard_image import ScorecardImage
from .bowling_analyzer import BowlingAnalyzer, BowlType
from .lbw_predictor import LbwPredictor
from .multi_cam_sync import MultiCameraSync, FfmpegNotFoundError
from .pro_comparison import ProComparison, PRO_PLAYER_DATABASE
