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

    def __init__(self, batting_hand="right", ball_color="red", fps=None):
        self.batting_hand = batting_hand
        self.ball_color = ball_color
        self.fps = fps

        # Components
        self.pose_estimator = PoseEstimator(
            static_mode=False,
            model_complexity=1,
            smooth=True,
        )
        self.ball_tracker = BallTracker(
            ball_color=ball_color,
            min_radius=2,
            max_radius=40,
            use_kalman=True,
        )
        self.bat_analyzer = BatAnalyzer(batting_hand=batting_hand)
        self.phase_detector = PhaseDetector(batting_hand=batting_hand, fps=fps or 30)
        self.metrics = MetricsCalculator(batting_hand=batting_hand, fps=fps or 30)
        self.visualizer = Visualizer(batting_hand=batting_hand)

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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Progress callback
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames, "Processing")

            # 1. Pose estimation
            pose_result = self.pose_estimator.process_frame(frame)

            # 2. Ball tracking
            ball_result = self.ball_tracker.track(frame, frame_idx)
            if ball_result["detected"]:
                ball_trajectory.append((ball_result["x"], ball_result["y"]))

            # 3. Bat analysis (from pose)
            bat_result = {}
            if pose_result["success"]:
                bat_result = self.bat_analyzer.analyze_swing(
                    pose_result["landmarks"], h, w, frame_idx
                )
                swing_path_history.append(bat_result.get("bat_tip"))

            # 4. Phase detection
            if pose_result["success"]:
                self.phase_detector.add_frame(pose_result["landmarks"], frame_idx)

            # 5. Frame metrics
            frame_metrics = {
                "frame": frame_idx,
                "timestamp_sec": frame_idx / max(1, video_fps),
            }

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

            # 6. Render output video
            if out_writer:
                # Draw overlays
                display = frame.copy()

                # Phase color overlay
                current_phase = self.phase_detector.get_phase_at_frame(frame_idx)
                if current_phase != BattingPhase.UNKNOWN.value:
                    display = self.visualizer.draw_phase_overlay(
                        display, current_phase, alpha=0.15
                    )
                    display = self.visualizer.draw_phase_text(display, current_phase, w)

                # Pose skeleton
                if pose_result["success"]:
                    display = self.pose_estimator.draw_landmarks(display, pose_result)

                # Ball trajectory
                if ball_trajectory:
                    display = self.visualizer.draw_ball_trajectory(
                        display, ball_trajectory[-50:], color=(0, 255, 255)
                    )
                if ball_result["detected"]:
                    cv2.circle(display, (ball_result["x"], ball_result["y"]),
                               ball_result.get("radius", 5) or 5,
                               (0, 255, 255), -1)

                # Bat swing
                if swing_path_history:
                    clean_path = [p for p in swing_path_history if p]
                    if clean_path:
                        display = self.visualizer.draw_swing_path(display, clean_path[-30:])

                # Bat line
                if bat_result.get("has_swing_data"):
                    display = self.visualizer.draw_bat_line(display, bat_result)

                # Metric HUD
                display = self.visualizer.draw_metric_hud(display, {
                    **frame_metrics,
                    "phase": current_phase,
                })

                # Frame counter
                cv2.putText(display, f"Frame {frame_idx}/{total_frames}",
                            (w - 180, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

        # Bat speed estimation
        bat_speed_kmh = self.bat_analyzer.estimate_bat_speed_kmh(max(1, video_fps))

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

            "bat_speed_kmh_estimate": bat_speed_kmh,
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
