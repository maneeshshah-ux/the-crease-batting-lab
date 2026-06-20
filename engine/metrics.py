"""
Metrics Calculator — Computes all cricket batting metrics from pose and ball tracking data.

Key metrics:
- Joint angles (elbow, knee, hip, shoulder)
- Head position & stability
- Weight transfer & balance
- Foot placement & stride
- Bat swing metrics (speed, angle, path)
- Ball trajectory & impact
- Timing analysis
"""

import numpy as np
import math


class MetricsCalculator:
    """
    Computes biomechanical and performance metrics for batting analysis.
    """

    def __init__(self, batting_hand="right", fps=30):
        self.batting_hand = batting_hand
        self.fps = fps

        if batting_hand == "right":
            self.front_shoulder = "LEFT_SHOULDER"
            self.back_shoulder = "RIGHT_SHOULDER"
            self.front_hip = "LEFT_HIP"
            self.back_hip = "RIGHT_HIP"
            self.front_knee = "LEFT_KNEE"
            self.back_knee = "RIGHT_KNEE"
            self.front_ankle = "LEFT_ANKLE"
            self.back_ankle = "RIGHT_ANKLE"
            self.front_elbow = "LEFT_ELBOW"
            self.back_elbow = "RIGHT_ELBOW"
            self.front_wrist = "LEFT_WRIST"
            self.back_wrist = "RIGHT_WRIST"
            self.front_foot = "LEFT_FOOT_INDEX"
            self.back_foot = "RIGHT_FOOT_INDEX"
            self.front_heel = "LEFT_HEEL"
            self.back_heel = "RIGHT_HEEL"
        else:
            self.front_shoulder = "RIGHT_SHOULDER"
            self.back_shoulder = "LEFT_SHOULDER"
            self.front_hip = "RIGHT_HIP"
            self.back_hip = "LEFT_HIP"
            self.front_knee = "RIGHT_KNEE"
            self.back_knee = "LEFT_KNEE"
            self.front_ankle = "RIGHT_ANKLE"
            self.back_ankle = "LEFT_ANKLE"
            self.front_elbow = "RIGHT_ELBOW"
            self.back_elbow = "LEFT_ELBOW"
            self.front_wrist = "RIGHT_WRIST"
            self.back_wrist = "LEFT_WRIST"
            self.front_foot = "RIGHT_FOOT_INDEX"
            self.back_foot = "LEFT_FOOT_INDEX"
            self.front_heel = "RIGHT_HEEL"
            self.back_heel = "LEFT_HEEL"

    @staticmethod
    def _angle_between(p1, p2, p3):
        """
        Calculate angle (in degrees) at p2 formed by p1-p2-p3.
        Points are dicts with 'pixel_x' and 'pixel_y'.
        """
        if not all([p1, p2, p3]):
            return None

        v1 = np.array([p1["pixel_x"] - p2["pixel_x"],
                       p1["pixel_y"] - p2["pixel_y"]])
        v2 = np.array([p3["pixel_x"] - p2["pixel_x"],
                       p3["pixel_y"] - p2["pixel_y"]])

        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm == 0:
            return None

        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _distance(p1, p2):
        """Euclidean distance between two landmark points."""
        if not p1 or not p2:
            return None
        return math.sqrt((p1["pixel_x"] - p2["pixel_x"])**2 +
                         (p1["pixel_y"] - p2["pixel_y"])**2)

    def compute_frame_metrics(self, landmarks):
        """
        Compute all metrics for a single frame.

        Args:
            landmarks: dict from PoseEstimator

        Returns dict of metric name -> value
        """
        m = {
            # Joint angles (degrees)
            "front_knee_angle": None,
            "back_knee_angle": None,
            "front_hip_angle": None,
            "back_hip_angle": None,
            "front_elbow_angle": None,
            "back_elbow_angle": None,
            "shoulder_angle": None,
            "hip_angle": None,
            "spine_angle": None,
            # Head
            "head_position_x": None,
            "head_position_y": None,
            "head_stability": None,  # relative to previous frame
            # Balance
            "stance_width": None,
            "weight_distribution": None,
            "hip_height": None,
            # Stride
            "stride_length": None,
            "foot_placement": None,
            # Timing proxy
            "hands_speed": 0,
        }

        # --- Joint angles ---
        # Front knee angle (hip-knee-ankle)
        m["front_knee_angle"] = self._angle_between(
            landmarks.get(self.front_hip),
            landmarks.get(self.front_knee),
            landmarks.get(self.front_ankle),
        )

        # Back knee angle
        m["back_knee_angle"] = self._angle_between(
            landmarks.get(self.back_hip),
            landmarks.get(self.back_knee),
            landmarks.get(self.back_ankle),
        )

        # Front elbow angle (shoulder-elbow-wrist)
        m["front_elbow_angle"] = self._angle_between(
            landmarks.get(self.front_shoulder),
            landmarks.get(self.front_elbow),
            landmarks.get(self.front_wrist),
        )

        # Back elbow angle
        m["back_elbow_angle"] = self._angle_between(
            landmarks.get(self.back_shoulder),
            landmarks.get(self.back_elbow),
            landmarks.get(self.back_wrist),
        )

        # Shoulder angle (horizontal tilt)
        l_s = landmarks.get("LEFT_SHOULDER")
        r_s = landmarks.get("RIGHT_SHOULDER")
        if l_s and r_s:
            shoulder_dy = l_s["pixel_y"] - r_s["pixel_y"]
            shoulder_dx = l_s["pixel_x"] - r_s["pixel_x"]
            m["shoulder_angle"] = float(np.degrees(np.arctan2(shoulder_dy, shoulder_dx)))

        # Hip angle (horizontal tilt)
        l_h = landmarks.get("LEFT_HIP")
        r_h = landmarks.get("RIGHT_HIP")
        if l_h and r_h:
            hip_dy = l_h["pixel_y"] - r_h["pixel_y"]
            hip_dx = l_h["pixel_x"] - r_h["pixel_x"]
            m["hip_angle"] = float(np.degrees(np.arctan2(hip_dy, hip_dx)))

        # Spine angle (angle of the back from vertical)
        nose = landmarks.get("NOSE")
        # Midpoint of hips
        mid_hip = None
        if l_h and r_h:
            mid_hip = {
                "pixel_x": (l_h["pixel_x"] + r_h["pixel_x"]) / 2,
                "pixel_y": (l_h["pixel_y"] + r_h["pixel_y"]) / 2,
            }
        if nose and mid_hip:
            spine_dx = nose["pixel_x"] - mid_hip["pixel_x"]
            spine_dy = nose["pixel_y"] - mid_hip["pixel_y"]
            # Angle from vertical
            m["spine_angle"] = float(np.degrees(np.arctan2(abs(spine_dx), spine_dy)))

        # --- Head position ---
        if nose:
            m["head_position_x"] = float(nose["pixel_x"])
            m["head_position_y"] = float(nose["pixel_y"])

        # --- Stance width ---
        l_ankle = landmarks.get("LEFT_ANKLE")
        r_ankle = landmarks.get("RIGHT_ANKLE")
        if l_ankle and r_ankle:
            m["stance_width"] = self._distance(l_ankle, r_ankle)

        # --- Hip height (proxy for knee bend) ---
        if l_h and r_h:
            m["hip_height"] = float((l_h["pixel_y"] + r_h["pixel_y"]) / 2)

        # --- Hands speed ---
        l_w = landmarks.get("LEFT_WRIST")
        r_w = landmarks.get("RIGHT_WRIST")
        if l_w and r_w:
            # Approximate hand speed from wrist movement between frames
            # (will be filled by temporal analysis)
            pass

        return m

    def compute_temporal_metrics(self, all_frame_metrics):
        """
        Compute time-series metrics across all frames.

        Args:
            all_frame_metrics: list of dicts from compute_frame_metrics

        Returns list of enhanced frame metrics with temporal info.
        """
        # Add velocity/speed information
        for i in range(len(all_frame_metrics)):
            if i > 0:
                prev = all_frame_metrics[i - 1]
                curr = all_frame_metrics[i]

                # Head movement
                if (curr.get("head_position_x") is not None and
                        prev.get("head_position_x") is not None):
                    head_dx = curr["head_position_x"] - prev["head_position_x"]
                    head_dy = curr["head_position_y"] - prev["head_position_y"]
                    curr["head_movement"] = float(np.sqrt(head_dx**2 + head_dy**2))
                else:
                    curr["head_movement"] = 0

            else:
                all_frame_metrics[i]["head_movement"] = 0

        return all_frame_metrics

    def compute_session_summary(self, all_metrics):
        """
        Compute aggregate statistics for an entire session.

        Args:
            all_metrics: list of per-frame metrics dicts

        Returns dict of summary statistics.
        """
        summary = {
            "total_frames": len(all_metrics),
            "duration_sec": round(len(all_metrics) / self.fps, 2) if self.fps else 0,
        }

        # Average joint angles
        angle_keys = ["front_knee_angle", "back_knee_angle",
                      "front_elbow_angle", "back_elbow_angle",
                      "shoulder_angle", "hip_angle", "spine_angle"]

        for key in angle_keys:
            values = [m[key] for m in all_metrics if m.get(key) is not None]
            if values:
                summary[f"avg_{key}"] = round(float(np.mean(values)), 1)
                summary[f"min_{key}"] = round(float(np.min(values)), 1)
                summary[f"max_{key}"] = round(float(np.max(values)), 1)
                summary[f"std_{key}"] = round(float(np.std(values)), 1)

        # Head stability (lower = more stable)
        head_movements = [m.get("head_movement", 0) for m in all_metrics
                          if m.get("head_movement") is not None]
        if head_movements:
            summary["avg_head_movement"] = round(float(np.mean(head_movements)), 2)
            summary["max_head_movement"] = round(float(np.max(head_movements)), 2)
            summary["head_stability_score"] = round(
                100 / (1 + float(np.mean(head_movements))), 1
            )

        # Stance width
        widths = [m["stance_width"] for m in all_metrics
                  if m.get("stance_width") is not None]
        if widths:
            summary["avg_stance_width"] = round(float(np.mean(widths)), 1)

        return summary

    def generate_coaching_tips(self, metrics_summary, phase_summary=None):
        """
        Generate automated coaching observations from metrics.

        Returns list of dicts: {category, observation, severity, suggestion}
        """
        tips = []

        # Front knee bend check
        if metrics_summary.get("avg_front_knee_angle"):
            avg = metrics_summary["avg_front_knee_angle"]
            if avg > 160:
                tips.append({
                    "category": "knee_bend",
                    "observation": "Front knee is too straight (%.0f°). Limited knee bend reduces power generation." % avg,
                    "severity": "medium",
                    "suggestion": "Work on getting your front knee more bent — aim for 130-150° at contact. This helps you get to the pitch of the ball and drive with power.",
                })
            elif avg < 120:
                tips.append({
                    "category": "knee_bend",
                    "observation": "Front knee is very bent (%.0f°). May cause you to be too low." % avg,
                    "severity": "low",
                    "suggestion": "You're getting low which is good for spin, but ensure you're not over-bending. Check balance at point of contact.",
                })

        # Head stability
        if metrics_summary.get("head_stability_score"):
            score = metrics_summary["head_stability_score"]
            if score < 40:
                tips.append({
                    "category": "head_position",
                    "observation": "Head movement is high (stability score: %.0f/100). Moving head compromises balance and shot timing." % score,
                    "severity": "high",
                    "suggestion": "Focus on keeping your head still and eyes level. Drill: Practice with a side-arm thrower and focus on watching the ball hit the bat without head movement.",
                })
            elif score < 70:
                tips.append({
                    "category": "head_position",
                    "observation": "Head stability is moderate (score: %.0f/100). Some unnecessary movement." % score,
                    "severity": "low",
                    "suggestion": "Good awareness. Try to consciously keep your head level when playing off front and back foot.",
                })

        # Backlift check
        if phase_summary:
            for shot in phase_summary:
                if shot.get("has_impact"):
                    tips.append({
                        "category": "shot_completed",
                        "observation": "Shot %d: %d phases detected, %.1fs duration." % (
                            shot["shot_number"], len(shot["phases"]), shot["duration_sec"]),
                        "severity": "info",
                        "suggestion": None,
                    })

        # Elbow angle
        if metrics_summary.get("avg_front_elbow_angle"):
            avg = metrics_summary["avg_front_elbow_angle"]
            if avg > 160:
                tips.append({
                    "category": "elbow_position",
                    "observation": "Front elbow is very straight (%.0f°). May limit your ability to control the bat through the line." % avg,
                    "severity": "medium",
                    "suggestion": "Keep the front elbow slightly bent (140-155°) at point of contact. This gives you better bat control and allows for softer hands.",
                })

        # Spine angle
        if metrics_summary.get("avg_spine_angle"):
            avg = metrics_summary["avg_spine_angle"]
            if avg < 10:
                tips.append({
                    "category": "posture",
                    "observation": "Spine is very upright (%.0f° from vertical). May cause you to play away from the body." % avg,
                    "severity": "medium",
                    "suggestion": "Bend slightly more from the waist. A 15-25° forward lean helps you get closer to the pitch of the ball and improves your drive.",
                })
            elif avg > 30:
                tips.append({
                    "category": "posture",
                    "observation": "Spine is leaning forward significantly (%.0f°). Balance may be compromised." % avg,
                    "severity": "low",
                    "suggestion": "Your forward lean is pronounced. Check that you can recover balance quickly — if you're falling over after the shot, straighten up slightly.",
                })

        return tips

    @staticmethod
    def angle_color(angle, ideal_min, ideal_max, warn_range=15):
        """Return a traffic-light color for an angle value."""
        if angle is None:
            return (128, 128, 128)  # grey
        if ideal_min <= angle <= ideal_max:
            return (0, 255, 0)  # green
        if (ideal_min - warn_range <= angle < ideal_min) or \
           (ideal_max < angle <= ideal_max + warn_range):
            return (0, 255, 255)  # yellow
        return (0, 0, 255)  # red
