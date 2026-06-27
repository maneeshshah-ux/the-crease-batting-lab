"""
Core Analysis Pipeline — Orchestrates pose estimation, ball tracking,
bat analysis, phase detection, and metrics calculation over a video file.
"""

import os
import json
import uuid
import cv2
import numpy as np
from datetime import datetime

from .pose_estimator import PoseEstimator
from .ball_tracker import BallTracker
from .bat_analyzer import BatAnalyzer
from .phase_detector import PhaseDetector, BattingPhase
from .metrics import MetricsCalculator
from .visualizer import Visualizer


class BattingAnalyser:
    """
    End-to-end batting session analyser.

    Usage:
        analyser = BattingAnalyser()
        result = analyser.analyse_video("path/to/video.mp4")
        # result contains all metrics, phase info, and summary
    """

    def __init__(self, batting_hand="right", ball_color="red", fps=None,
                 frame_step=2, camera_view="side_off"):
        self.batting_hand = batting_hand
        self.ball_color = ball_color
        self.fps = fps
        self.frame_step = max(1, frame_step)
        self.camera_view = camera_view

        # Components
        self.pose_estimator = PoseEstimator(
            static_mode=False,
            model_complexity=0,  # lite model: 3x faster, 1/3 memory
            smooth=True,
        )
        self.ball_tracker = BallTracker(
            ball_color=ball_color,
            min_radius=2,
            max_radius=40,
            use_kalman=True,
        )
        self.bat_analyzer = BatAnalyzer(batting_hand=batting_hand,
                                        camera_view=camera_view)
        self.phase_detector = PhaseDetector(batting_hand=batting_hand, fps=fps or 30)
        self.metrics = MetricsCalculator(batting_hand=batting_hand, fps=fps or 30,
                                         camera_view=camera_view)
        self.visualizer = Visualizer(batting_hand=batting_hand)
        self.calibration_px_per_m = None  # set after analysis loop

    def analyse_video(self, video_path, output_dir=None, generate_video=True,
                      progress_callback=None):
        """
        Run full analysis on a video file.

        Args:
            video_path: Path to video file
            output_dir: Where to save results (default: alongside video)
            generate_video: Whether to render annotated output video
            progress_callback: Optional fn(frame, total, status)

        Returns dict with full analysis results.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": f"Cannot open video: {video_path}"}

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps is None:
            self.fps = video_fps if video_fps > 0 else 30
            self.phase_detector.fps = self.fps
            self.metrics.fps = self.fps

        # Prepare output
        session_id = str(uuid.uuid4())[:8]
        if output_dir is None:
            output_dir = os.path.dirname(video_path) or "."

        os.makedirs(output_dir, exist_ok=True)

        # Output video writer
        if generate_video:
            output_video_path = os.path.join(output_dir, f"analysis_{session_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 — 5-10x smaller than mp4v
            out_writer = cv2.VideoWriter(output_video_path, fourcc,
                                         max(1, video_fps), (w, h))
        else:
            output_video_path = None
            out_writer = None

        # Data collectors
        all_frame_metrics = []
        ball_trajectory = []
        phase_labels = []
        bat_speed_history = []
        joint_histories = {
            "front_knee": [],
            "back_knee": [],
            "front_elbow": [],
            "back_elbow": [],
            "spine_angle": [],
            "shoulder_angle": [],
        }
        head_movement_history = []
        swing_path_history = []

        frame_idx = 0
        _prev_display = None  # cache last annotated frame for skipped frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Progress callback
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames, "Processing")

            # ── Frame step: skip heavy analysis every Nth frame ──────────────
            should_analyze = (frame_idx % self.frame_step == 0)

            # 1. Pose estimation (with wicketkeeper filtering)
            if should_analyze:
                pose_result = self.pose_estimator.process_frame(frame, prefer_batter=True)
                # Skip analysis if detected person is likely not the batter
                if pose_result["success"] and not pose_result.get("is_batter", True):
                    pose_result["success"] = False
            else:
                pose_result = {"success": False, "landmarks": {},
                               "landmark_list": None, "raw": None, "is_batter": False}

            # 2. Ball tracking (runs every frame — cheap)
            ball_result = self.ball_tracker.track(frame, frame_idx)
            if ball_result["detected"]:
                ball_trajectory.append((ball_result["x"], ball_result["y"]))

            # 3. Bat analysis (from pose — only on analyzed frames)
            bat_result = {}
            if pose_result["success"] and should_analyze:
                bat_result = self.bat_analyzer.analyze_swing(
                    pose_result["landmarks"], h, w, frame_idx
                )
                swing_path_history.append(bat_result.get("bat_tip"))

            # 4. Phase detection
            if pose_result["success"] and should_analyze:
                self.phase_detector.add_frame(pose_result["landmarks"], frame_idx)

            # 5. Frame metrics — only store for analyzed frames
            if should_analyze:
                frame_metrics = {
                    "frame": frame_idx,
                    "timestamp_sec": frame_idx / max(1, video_fps),
                }

                fm = {}  # default empty
                if pose_result["success"]:
                    fm = self.metrics.compute_frame_metrics(pose_result["landmarks"])
                    frame_metrics.update(fm)

                # Add bat metrics
                frame_metrics["bat_speed_px"] = bat_result.get("bat_speed_px", 0)
                frame_metrics["bat_angle_deg"] = bat_result.get("bat_angle_deg")
                frame_metrics["bat_lift_height"] = bat_result.get("bat_lift_height")

                # Add ball info
                frame_metrics["ball_detected"] = ball_result["detected"]
                frame_metrics["ball_x"] = ball_result["x"]
                frame_metrics["ball_y"] = ball_result["y"]

                all_frame_metrics.append(frame_metrics)

                # Collect time series
                if fm.get("front_knee_angle") is not None:
                    joint_histories["front_knee"].append(fm["front_knee_angle"])
                if fm.get("back_knee_angle") is not None:
                    joint_histories["back_knee"].append(fm["back_knee_angle"])
                if fm.get("front_elbow_angle") is not None:
                    joint_histories["front_elbow"].append(fm["front_elbow_angle"])
                if fm.get("back_elbow_angle") is not None:
                    joint_histories["back_elbow"].append(fm["back_elbow_angle"])
                if fm.get("spine_angle") is not None:
                    joint_histories["spine_angle"].append(fm["spine_angle"])
                if fm.get("shoulder_angle") is not None:
                    joint_histories["shoulder_angle"].append(fm["shoulder_angle"])
                if fm.get("head_movement") is not None:
                    head_movement_history.append(fm["head_movement"])
                if bat_result.get("bat_speed_px", 0) > 0:
                    bat_speed_history.append(bat_result["bat_speed_px"])

            # 6. Render output video (only annotate analyzed frames)
            if out_writer:
                display = frame.copy()
                if should_analyze and pose_result["success"]:
                    current_phase = self.phase_detector.get_phase_at_frame(frame_idx)

                    # --- Phase bar at top ---
                    display = self.visualizer.draw_phase_bar(display, current_phase)

                    # --- Head stability indicator (traffic light) ---
                    nose_lm = pose_result["landmarks"].get("NOSE", {})
                    head_x = nose_lm.get("pixel_x") if nose_lm else None
                    head_y = nose_lm.get("pixel_y") if nose_lm else None
                    head_mvmt = fm.get("head_movement", 0) if fm else 0

                    display = self.visualizer.draw_head_indicator(
                        display, head_mvmt, head_x, head_y
                    )

                    # --- Balance — spirit level ---
                    display = self.visualizer.draw_balance_level(
                        display, fm.get("spine_angle", None), fm.get("front_knee_angle", None)
                    )

                    # --- Bat swing path ---
                    if swing_path_history:
                        clean_path = [p for p in swing_path_history if p]
                        if clean_path:
                            display = self.visualizer.draw_swing_path(
                                display, clean_path[-30:], phase=current_phase
                            )

                    # --- Bat line ---
                    if bat_result.get("has_swing_data"):
                        display = self.visualizer.draw_bat_line(display, bat_result)

                    # --- Bat speed — speedometer ---
                    bat_spd_px = fm.get("bat_speed_px", 0)
                    px_per_m_live = (self.calibration_px_per_m or 120.0)
                    lever = self.bat_analyzer.HAND_TO_TIP_FACTOR
                    live_kmh = (bat_spd_px * video_fps / px_per_m_live * 3.6 * lever
                               if bat_spd_px > 0 else None)
                    if live_kmh and live_kmh > 200:
                        live_kmh = None
                    if live_kmh and live_kmh > (self.visualizer.session_peak_kmh or 0):
                        self.visualizer.session_peak_kmh = live_kmh
                    display = self.visualizer.draw_speedometer(
                        display, speed_kmh=live_kmh,
                        calibration_available=self.calibration_px_per_m is not None,
                        peak_session_kmh=self.visualizer.session_peak_kmh,
                    )

                    # --- Bat speed overlay ---
                    if live_kmh and live_kmh > 5:
                        display = self.visualizer.draw_bat_speed_overlay(
                            display, live_kmh, impact_frame=(current_phase == "impact")
                        )

                    # --- Weight transfer ---
                    display = self.visualizer.draw_weight_transfer(
                        display, pose_result["landmarks"]
                    )

                    # --- Phase legend ---
                    if frame_idx < 10 or frame_idx % 600 == 0:
                        display = self.visualizer.draw_phase_legend(display)

                    _prev_display = display  # cache for skipped frames
                elif _prev_display is not None:
                    # Skipped frame: reuse last annotated display
                    display = _prev_display

                # --- Watermark (every frame) ---
                display = self.visualizer.draw_watermark(display)
                out_writer.write(display)

            frame_idx += 1

        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()

        # --- Post-processing ---

        # Run phase detection across all frames
        phases = self.phase_detector.analyze_phases(self.bat_analyzer)
        shot_summary = self.phase_detector.get_shot_summary()

        # Compute temporal metrics
        all_frame_metrics = self.metrics.compute_temporal_metrics(all_frame_metrics)

        # Session summary
        session_summary = self.metrics.compute_session_summary(all_frame_metrics)

        # Ball speed estimation
        ball_speed = self.ball_tracker.estimate_speed(max(1, video_fps))

        # Coaching tips
        coaching_tips = self.metrics.generate_coaching_tips(
            session_summary, shot_summary
        )

        # Auto-calibrate from all collected landmarks
        all_landmarks = [fd["landmarks"] for fd in self.phase_detector.frame_data]
        cal_result = self.bat_analyzer.calibrate_from_landmarks(
            all_landmarks, h, w
        )

        # Pass calibration to visualizer for real-world units (cm)
        if cal_result and cal_result.get("px_per_m"):
            self.calibration_px_per_m = cal_result["px_per_m"]
            self.visualizer.px_per_cm = cal_result["px_per_m"] / 100.0

        # Bat speed estimation (now calibrated to km/h)
        # Build full-frame speed array for continuity-based spike filtering
        if len(all_frame_metrics) > 5:
            full_speed_array = [m.get("bat_speed_px", 0) for m in all_frame_metrics]
            bat_speed_kmh = self.bat_analyzer.estimate_bat_speed_kmh(
                max(1, video_fps),
                speeds=full_speed_array  # full frame-sequential array
            )
        else:
            bat_speed_kmh = self.bat_analyzer.estimate_bat_speed_kmh(
                max(1, video_fps)
            )

        # Log spike info
        n_outliers = bat_speed_kmh.get("outliers_removed", 0)
        if n_outliers:
            print(f"  Filtered {n_outliers} outlier bat speed readings")

        # Backlift peak
        backlift_peak = self.bat_analyzer.detect_backlift_peak()

        # Build final result
        result = {
            "success": True,
            "session_id": session_id,
            "video_path": video_path,
            "output_video_path": output_video_path,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / max(1, video_fps), 2),
            "video_fps": video_fps,
            "frame_width": w,
            "frame_height": h,
            "batting_hand": self.batting_hand,
            "ball_color": self.ball_color,
            "analysis_timestamp": datetime.now().isoformat(),

            "phases": phases,
            "shot_summary": shot_summary,
            "num_shots_detected": len(shot_summary),

            "ball_trajectory_length": len(ball_trajectory),
            "ball_speed": ball_speed,

            "session_summary": session_summary,
            "coaching_tips": coaching_tips,

            "bat_speed": bat_speed_kmh,
            "backlift_peak": backlift_peak,

            # Full metrics per frame (sampled for large videos)
            "frame_metrics": all_frame_metrics[::max(1, total_frames // 500)] if total_frames > 500 else all_frame_metrics,
            "frame_metrics_full": len(all_frame_metrics),

            # Histories for charting
            "joint_histories": {
                k: v[::max(1, len(v) // 300)] if len(v) > 300 else v
                for k, v in joint_histories.items()
            },
            "bat_speed_history": bat_speed_history[::max(1, len(bat_speed_history) // 200)] if len(bat_speed_history) > 200 else bat_speed_history,
            "head_movement_history": head_movement_history[::max(1, len(head_movement_history) // 200)] if len(head_movement_history) > 200 else head_movement_history,
        }

        # Generate charts
        result["charts"] = self.visualizer.create_session_charts({
            **joint_histories,
            "bat_speed_history": bat_speed_history,
            "head_movement_history": head_movement_history,
        })

        # Save result to JSON
        result_path = os.path.join(output_dir, f"analysis_{session_id}.json")
        with open(result_path, "w") as f:
            # Convert numpy types for JSON
            json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else (
                int(x) if isinstance(x, (np.integer,)) else str(x)
            ))

        result["result_path"] = result_path
        return result

    def compare_sessions(self, session_results):
        """
        Compare multiple analysis results.

        Args:
            session_results: list of analysis result dicts

        Returns comparison dict.
        """
        comparison = {
            "num_sessions": len(session_results),
            "metrics_comparison": {},
        }

        if len(session_results) < 2:
            return comparison

        key_metrics = ["avg_front_knee_angle", "avg_back_knee_angle",
                       "avg_front_elbow_angle", "avg_spine_angle",
                       "avg_head_movement", "head_stability_score"]

        for metric in key_metrics:
            values = []
            for s in session_results:
                summary = s.get("session_summary", {})
                if metric in summary:
                    values.append(summary[metric])

            if values:
                comparison["metrics_comparison"][metric] = {
                    "values": values,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "improvement": float(values[-1] - values[0]) if len(values) >= 2 else 0,
                }

        return comparison

    def close(self):
        """Clean up resources."""
        self.pose_estimator.close()
