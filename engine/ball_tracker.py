"""
Front-on Ball Tracker — 4-phase ball detection and trajectory tracking.

Camera position:   Non-striker end, behind bowler, looking toward batter.
Ball behaviour:    Starts large (8-15 px at release), gets tiny (1-3 px at batter).
                   Track robustly for ~15 frames, then Kalman-extrapolate.

Tracking phases (state machine):
  1. IDLE     — Waiting for bowler_delivery person_label
  2. RELEASE  — Color-based detection in upper-central ROI. Ball 8-15 px.
  3. FLIGHT   — Motion differencing + color along predicted corridor. Ball 3-8 px.
               Pitch-point detected as velocity kink in trajectory.
  4. APPROACH — Kalman extrapolation only (ball too small for visual). Ball 1-3 px.
               Predicts line at batter's crease.
  5. IMPACT   — Dormant; impact is detected externally by bat_speed / audio.
  6. DONE     — Tracking complete; reset on next idle cycle.

Speed estimation:   From pixel displacement in first ~15 frames with calibration.
Pitch detection:    From velocity-angle change in trajectory (>30° = bounce event).
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────

RELEASE_MAX_FRAMES = 12        # release phase lasts at most 12 frames
FLIGHT_MAX_FRAMES = 40         # flight phase transitions to approach at frame 40
APPROACH_MAX_FRAMES = 60       # approach transitions to impact at frame 60

RELEASE_ROI_MARGIN = 0.15      # fraction of frame dim around expected release area
FLIGHT_CORRIDOR_WIDTH = 0.12   # fraction of frame width for flight search band
PITCH_ANGLE_THRESHOLD = 30.0   # degrees — trajectory direction change = pitch event
MIN_SPEED_FRAMES = 5           # minimum frames for speed estimate
MIN_PITCH_FRAMES = 8           # minimum frames before we look for a pitch point

# Cricket ball HSV ranges
BALL_COLOR_RANGES = {
    "red": {
        "lower1": np.array([0, 80, 80]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 80, 80]),
        "upper2": np.array([180, 255, 255]),
    },
    "pink": {
        "lower": np.array([160, 40, 180]),
        "upper": np.array([180, 100, 255]),
    },
    "white": {
        "lower": np.array([0, 0, 180]),
        "upper": np.array([180, 40, 255]),
    },
    "tennis": {
        "lower": np.array([30, 100, 100]),
        "upper": np.array([80, 255, 255]),
    },
}


class TrackingPhase(Enum):
    """Ball tracking state machine phases."""
    IDLE = "idle"
    RELEASE = "release"
    FLIGHT = "flight"
    APPROACH = "approach"
    IMPACT = "impact"
    DONE = "done"


# ────────────────────────────────────────────────────────────────────────
# Kalman Filter
# ────────────────────────────────────────────────────────────────────────

class KalmanFilter2D:
    """Simple 2D Kalman filter for ball position smoothing and prediction.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.5):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False

    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = True

    def predict(self) -> Optional[Tuple[int, int]]:
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return (int(pred[0]), int(pred[1]))

    def update(self, x: float, y: float) -> Tuple[int, int]:
        if not self.initialized:
            self.init(x, y)
            return (int(x), int(y))
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measured)
        return (int(corrected[0]), int(corrected[1]))

    def reset(self):
        self.initialized = False


# ────────────────────────────────────────────────────────────────────────
# Front-on Ball Tracker
# ────────────────────────────────────────────────────────────────────────

class FrontOnBallTracker:
    """4-phase front-on ball tracker with person-label awareness.

    Only searches for the ball when the PersonTracker reports
    ``bowler_delivery`` or ``bowler_follow_through``.  At other times
    it stays in IDLE to avoid false detections from bat / hands / netting.
    """

    def __init__(
        self,
        ball_color: str = "red",
        fps: float = 30.0,
        calibration_px_per_m: Optional[float] = None,
        min_ball_radius: int = 2,
        max_ball_radius: int = 40,
        # Backward-compatible aliases
        min_radius: Optional[int] = None,
        max_radius: Optional[int] = None,
        use_kalman: Optional[bool] = None,
    ):
        if min_radius is not None:
            min_ball_radius = min_radius
        if max_radius is not None:
            max_ball_radius = max_radius
        self.ball_color = ball_color
        self.fps = fps
        self.calibration_px_per_m = calibration_px_per_m
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius

        # State
        self.phase = TrackingPhase.IDLE
        self.kalman = KalmanFilter2D()
        self.trajectory: List[Tuple[int, float, float]] = []   # (frame_idx, x, y)
        self.raw_detections: List[Tuple[int, float, float, float]] = []  # (frame, x, y, conf)
        self.release_point: Optional[Tuple[float, float]] = None
        self.release_frame: Optional[int] = None
        self.pitch_point: Optional[Tuple[float, float]] = None
        self.pitch_frame: Optional[int] = None
        self.bowler_start_frame: Optional[int] = None
        self.last_detection_frame: Optional[int] = None
        self.consecutive_misses = 0

        # For motion-based detection
        self._prev_gray: Optional[np.ndarray] = None

        # Speed estimate (computed once after collection)
        self._speed_kmh: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────────────

    def track(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_label: str = "empty",
    ) -> Dict[str, Any]:
        """Process a single frame and return ball tracking results.

        Args:
            frame: BGR frame from video.
            frame_idx: Zero-based frame index.
            person_label: Current label from ``PersonTracker.process_frame()``.

        Returns:
            dict with keys:
                detected (bool), x (int|None), y (int|None),
                radius (int|None), confidence (float),
                phase (str), trajectory (list of (x,y)),
                kalman_predicted ((int,int)|None),
                release_point ((float,float)|None), release_frame (int|None),
                pitch_point ((float,float)|None), pitch_frame (int|None),
                speed_kmh (float|None).
        """
        h, w = frame.shape[:2]

        # ── State machine ──────────────────────────────────────────────
        self._update_machine(person_label, frame_idx)

        # ── Build result skeleton ──────────────────────────────────────
        result = self._empty_result()

        if self.phase == TrackingPhase.IDLE:
            return result

        # ── Frame count since bowler delivery ──────────────────────────
        frames_since_start = frame_idx - self.bowler_start_frame \
            if self.bowler_start_frame is not None else 0

        # ── Advance phase by frame count ───────────────────────────────
        self._advance_phase_by_frame(frames_since_start)

        # ── Kalman predict first (always advance temporal state) ───────
        kalman_prediction = self.kalman.predict()

        # ── Detection strategy per phase ───────────────────────────────
        ball_data = None  # (x, y, radius, confidence)

        if self.phase == TrackingPhase.RELEASE:
            ball_data = self._detect_in_release(frame, h, w, frames_since_start)

        elif self.phase == TrackingPhase.FLIGHT:
            ball_data = self._detect_in_flight(frame, h, w)
            # If nothing found in flight, try a wider colour search
            if ball_data is None:
                ball_data = self._color_detect_full(frame, h, w)

        elif self.phase == TrackingPhase.APPROACH:
            # Ball too small for visual; try colour search but mainly
            # rely on Kalman extrapolation
            ball_data = self._color_detect_full(frame, h, w, conf_penalty=0.5)
            if ball_data is None and kalman_prediction is not None:
                ball_data = (*kalman_prediction, None, 0.25)

        # ── Update Kalman & trajectory ─────────────────────────────────
        if ball_data is not None:
            bx, by, bradius, conf = ball_data
            bx_i, by_i = int(round(bx)), int(round(by))
            self.consecutive_misses = 0

            # Kalman correct (update with measurement)
            smoothed = self.kalman.update(bx_i, by_i)
            sx, sy = smoothed

            self.trajectory.append((frame_idx, float(sx), float(sy)))
            self.raw_detections.append((frame_idx, float(bx_i), float(by_i), conf))
            self.last_detection_frame = frame_idx

            # Capture release point (first successful detection within RELEASE phase)
            if self.release_point is None and frames_since_start < RELEASE_MAX_FRAMES:
                self.release_point = (float(sx), float(sy))
                self.release_frame = frame_idx

            result["detected"] = True
            result["x"] = sx
            result["y"] = sy
            result["radius"] = bradius
            result["confidence"] = round(conf, 3)

        else:
            self.consecutive_misses += 1
            # Use Kalman prediction to keep trajectory alive
            if kalman_prediction is not None:
                self.trajectory.append((frame_idx, float(kalman_prediction[0]),
                                        float(kalman_prediction[1])))
                result["kalman_predicted"] = kalman_prediction

            # Bail: too many misses in a row → done
            if self.consecutive_misses > 15 and self.phase != TrackingPhase.IDLE:
                self.phase = TrackingPhase.DONE

        # ── Detect pitch point ─────────────────────────────────────────
        if self.pitch_point is None and len(self.trajectory) >= MIN_PITCH_FRAMES:
            self._detect_pitch_point()

        # ── Speed estimation ───────────────────────────────────────────
        if self.release_point is not None and len(self.trajectory) >= MIN_SPEED_FRAMES:
            self._estimate_speed()
            result["speed_kmh"] = self._speed_kmh

        # ── Finalise result ────────────────────────────────────────────
        result["phase"] = self.phase.value
        result["trajectory"] = [(int(x), int(y)) for (_, x, y) in self.trajectory]
        result["release_point"] = self.release_point
        result["release_frame"] = self.release_frame
        result["pitch_point"] = self.pitch_point
        result["pitch_frame"] = self.pitch_frame

        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return result

    def reset(self):
        """Reset tracker for a new delivery."""
        self.phase = TrackingPhase.IDLE
        self.kalman.reset()
        self.trajectory.clear()
        self.raw_detections.clear()
        self.release_point = None
        self.release_frame = None
        self.pitch_point = None
        self.pitch_frame = None
        self.bowler_start_frame = None
        self.last_detection_frame = None
        self.consecutive_misses = 0
        self._prev_gray = None
        self._speed_kmh = None

    # ── State machine helpers ──────────────────────────────────────────

    def _update_machine(self, person_label: str, frame_idx: int):
        """Transition the state machine based on person_label."""
        if person_label.startswith("bowler") and self.phase == TrackingPhase.IDLE:
            # Start tracking: bowler is active
            self.phase = TrackingPhase.RELEASE
            self.bowler_start_frame = frame_idx
            self.consecutive_misses = 0
            self.kalman.reset()
            self._prev_gray = None

        elif person_label == "batter" and self.phase not in (TrackingPhase.IDLE, TrackingPhase.DONE):
            # Batter appeared → bowler delivery has passed
            # (do nothing abrupt; let frame-count transitions handle it)

            # If we haven't detected anything yet and batter is back, give up
            if self.last_detection_frame is None:
                self.phase = TrackingPhase.DONE

        elif person_label == "empty" and self.phase == TrackingPhase.DONE:
            # Reset fully when the frame is empty after a completed delivery
            pass  # Don't self-reset; let caller call reset() explicitly

    def _advance_phase_by_frame(self, frames_since_start: int):
        """Time-based phase transitions."""
        if self.phase == TrackingPhase.RELEASE and frames_since_start >= RELEASE_MAX_FRAMES:
            self.phase = TrackingPhase.FLIGHT
        elif self.phase == TrackingPhase.FLIGHT and frames_since_start >= FLIGHT_MAX_FRAMES:
            self.phase = TrackingPhase.APPROACH
        elif self.phase == TrackingPhase.APPROACH and frames_since_start >= APPROACH_MAX_FRAMES:
            self.phase = TrackingPhase.IMPACT

    # ── Detection strategies ───────────────────────────────────────────

    def _detect_in_release(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        frames_since_start: int,
    ) -> Optional[Tuple[float, float, Optional[int], float]]:
        """Colour-based detection in the release zone (upper frame centre).

        Returns (x, y, radius, confidence) or None.
        """
        # Release ROI: upper central portion of frame
        x_center = w // 2
        roi_w = int(w * RELEASE_ROI_MARGIN * 2)
        roi_h = int(h * RELEASE_ROI_MARGIN * 2)
        x1 = max(0, x_center - roi_w // 2)
        y1 = max(0, 0)
        x2 = min(w, x_center + roi_w // 2)
        y2 = min(h, roi_h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        result = self._detect_ball_in_roi(roi, h, w, x_offset=x1, y_offset=y1)

        # If no result yet, widen search to full frame after a few frames
        if result is None and frames_since_start > 5:
            result = self._color_detect_full(frame, h, w)

        return result

    def _detect_in_flight(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
    ) -> Optional[Tuple[float, float, Optional[int], float]]:
        """Motion-based detection in a corridor from release to batter.

        Returns (x, y, radius, confidence) or None.
        """
        # Use corridor if we have a trajectory, otherwise full frame
        if len(self.trajectory) >= 2:
            # Predict next position from last two points
            *_, last_x, last_y = self.trajectory[-1]
            if len(self.trajectory) >= 2:
                *_, prev_x, prev_y = self.trajectory[-2]
            else:
                prev_x, prev_y = last_x, last_y

            dx = last_x - prev_x
            dy = last_y - prev_y
            pred_x = int(last_x + dx)
            pred_y = int(last_y + dy)

            # Search ROI: corridor around predicted position
            margin = int(w * FLIGHT_CORRIDOR_WIDTH)
            x1 = max(0, pred_x - margin)
            y1 = max(0, pred_y - margin)
            x2 = min(w, pred_x + margin)
            y2 = min(h, pred_y + margin)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return self._color_detect_full(frame, h, w)

            result = self._detect_ball_in_roi(roi, h, w, x_offset=x1, y_offset=y1)

            # Also try motion detection in same ROI
            if result is None and self._prev_gray is not None:
                prev_roi = self._prev_gray[y1:y2, x1:x2]
                curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(prev_roi, curr_gray)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                best = None
                best_score = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 10:
                        continue
                    ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
                    if radius < self.min_ball_radius or radius > self.max_ball_radius:
                        continue
                    # Preference: small, circular motion blobs
                    perim = cv2.arcLength(cnt, True)
                    circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
                    score = min(area / 50.0, 3.0) * (0.5 + 0.5 * circ)
                    if score > best_score:
                        best_score = score
                        best = (cx + x1, cy + y1, int(radius), score)

                if best is not None:
                    return best

            return result
        else:
            return self._color_detect_full(frame, h, w)

    def _detect_ball_in_roi(
        self,
        roi: np.ndarray,
        frame_h: int,
        frame_w: int,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> Optional[Tuple[float, float, Optional[int], float]]:
        """Colour-based ball detection within a region of interest.

        Returns (x, y, radius, confidence) in original frame coordinates,
        or None if nothing found.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._create_color_mask(hsv)

        if mask is None or mask.sum() == 0:
            return None

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < 0.3:  # relaxed for small balls
                continue

            ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
            if radius < self.min_ball_radius or radius > self.max_ball_radius:
                continue

            # Score: prefer circular and reasonably sized
            score = circularity * min(radius / 10.0, 3.0)
            if score > best_score:
                best_score = score
                best = (cx + x_offset, cy + y_offset, int(radius), score)

        return best

    def _color_detect_full(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        conf_penalty: float = 1.0,
    ) -> Optional[Tuple[float, float, Optional[int], float]]:
        """Colour-based ball detection across the entire frame.

        Returns (x, y, radius, confidence * conf_penalty) or None.
        """
        result = self._detect_ball_in_roi(frame, h, w)
        if result is not None:
            x, y, r, conf = result
            return (x, y, r, conf * conf_penalty)
        return None

    def _create_color_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """Create HSV colour mask for the configured ball colour."""
        ranges = BALL_COLOR_RANGES.get(self.ball_color)
        if ranges is None:
            return None

        if self.ball_color == "red":
            mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv, ranges["lower"], ranges["upper"])

    # ── Pitch point detection ──────────────────────────────────────────

    def _detect_pitch_point(self):
        """Detect pitch point from trajectory.

        Approach:
          1. Look for a "kink" where the direction-angle between
             successive velocity vectors exceeds PITCH_ANGLE_THRESHOLD.
          2. Also look for a sharp y-velocity change (ball slowing
             or reversing vertical direction after bounce).

        Uses raw (pre-Kalman) positions for sharper angle detection
        but maps the result back to trajectory frame indices.
        """
        if len(self.trajectory) < MIN_PITCH_FRAMES:
            return

        # Use raw detections for velocity calculation (sharper kinks),
        # but map back to trajectory for frame indices.
        if len(self.raw_detections) >= MIN_PITCH_FRAMES:
            raw_positions = np.array([(x, y) for (_, x, y, _) in self.raw_detections])
            raw_frames = [f for (f, _, _, _) in self.raw_detections]
        else:
            raw_positions = np.array([(x, y) for (_, x, y) in self.trajectory])
            raw_frames = [f for (f, _, _) in self.trajectory]

        velocities = np.diff(raw_positions, axis=0)
        if len(velocities) < 2:
            return

        # Strategy 1: direction-angle kink
        angles = []
        for i in range(1, len(velocities)):
            v1 = velocities[i - 1]
            v2 = velocities[i]
            dot = float(v1[0] * v2[0] + v1[1] * v2[1])
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                angles.append(0.0)
                continue
            cos_angle = np.clip(dot / (n1 * n2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        max_angle = max(angles) if angles else 0

        # Strategy 2: y-velocity change (vertical bounce signature)
        vy = velocities[:, 1]
        vy_changes = np.abs(np.diff(vy))
        max_vy_change = float(np.max(vy_changes)) if len(vy_changes) > 0 else 0

        # Pick the best pitch frame from candidates
        candidates = []

        # Option A: direction kink
        if max_angle > PITCH_ANGLE_THRESHOLD:
            idx = angles.index(max_angle)
            # Velocity idx is between raw_positions[idx] and raw_positions[idx+1]
            # Pitch frame is raw_frames[idx+1]
            pitch_frame = raw_frames[idx + 1] if (idx + 1) < len(raw_frames) else None
            if pitch_frame is not None:
                candidates.append((pitch_frame, max_angle / 90.0))

        # Option B: y-velocity change
        if max_vy_change > 5.0:
            vy_idx = int(np.argmax(vy_changes))
            # vy_changes[vy_idx] is between velocity[vy_idx] and velocity[vy_idx+1]
            # Which involves raw_frames[vy_idx+1] and raw_frames[vy_idx+2]
            pitch_frame = raw_frames[vy_idx + 2] if (vy_idx + 2) < len(raw_frames) else None
            if pitch_frame is not None:
                candidates.append((pitch_frame, max_vy_change / 20.0))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            best_frame, _ = candidates[0]
            # Find corresponding position in trajectory
            for f, x, y in self.trajectory:
                if f == best_frame:
                    self.pitch_point = (float(x), float(y))
                    self.pitch_frame = int(f)
                    break

    # ── Speed estimation ───────────────────────────────────────────────

    def _estimate_speed(self) -> Optional[float]:
        """Estimate ball speed from trajectory data.

        Uses the first N frames of trajectory where the ball is closest
        to the camera (largest apparent size / highest confidence).
        Converts pixel displacement to km/h using calibration.

        Cache result in ``self._speed_kmh``.
        """
        if self._speed_kmh is not None:
            return self._speed_kmh

        if len(self.trajectory) < MIN_SPEED_FRAMES:
            return None

        # Use the first detected portion (release and early flight)
        # where ball is most visible and measurement most reliable
        positions = np.array([(x, y) for (_, x, y) in self.trajectory[:15]])

        if len(positions) < 3:
            return None

        # Pixel displacement: total distance traveled
        deltas = np.diff(positions, axis=0)
        distances = np.sqrt(deltas[:, 0]**2 + deltas[:, 1]**2)
        total_px = float(np.sum(distances))

        # Time span in seconds
        if len(self.trajectory) >= 2:
            t_start = self.trajectory[0][0]
            t_end = self.trajectory[min(len(self.trajectory) - 1, 14)][0]
            dt = max(1, t_end - t_start) / self.fps
        else:
            return None

        if dt <= 0:
            return None

        px_per_sec = total_px / dt

        # Convert to km/h
        if self.calibration_px_per_m and self.calibration_px_per_m > 0:
            speed_ms = px_per_sec / self.calibration_px_per_m
            speed_kmh = speed_ms * 3.6
        else:
            # Without calibration, store px/s and let caller convert
            speed_kmh = None

        self._speed_kmh = speed_kmh
        return speed_kmh

    # ── Public speed estimation (backward-compatible) ─────────────────

    def estimate_speed(self, fps: Optional[float] = None) -> Dict[str, Any]:
        """Estimate ball speed from trajectory.

        Backward-compatible wrapper that returns the same dict structure
        as the original ``BallTracker.estimate_speed()``.

        Args:
            fps: Frames per second (uses ``self.fps`` if not provided).

        Returns:
            dict with keys: estimated, speed_kmh, speed_px_per_frame, ...
        """
        if fps is not None:
            self.fps = fps

        self._estimate_speed()

        if self._speed_kmh is not None:
            return {
                "estimated": True,
                "speed_kmh": self._speed_kmh,
                "speed_px_per_frame": 0,
                "speed_px_per_sec": 0,
            }

        # Fallback: pixel-based estimate
        if len(self.trajectory) >= 2:
            positions = np.array([(x, y) for (_, x, y) in self.trajectory[:15]])
            if len(positions) >= 3:
                deltas = np.diff(positions, axis=0)
                distances = np.sqrt(deltas[:, 0]**2 + deltas[:, 1]**2)
                avg_speed_px = float(np.mean(distances))
                return {
                    "estimated": True,
                    "speed_kmh": None,
                    "speed_px_per_frame": avg_speed_px,
                    "speed_px_per_sec": avg_speed_px * self.fps,
                }

        return {
            "estimated": False,
            "speed_kmh": 0,
            "speed_px_per_frame": 0,
            "speed_px_per_sec": 0,
        }

    # ── Misc helpers ───────────────────────────────────────────────────

    def _empty_result(self) -> Dict[str, Any]:
        """Return a default empty result dict."""
        return {
            "detected": False,
            "x": None,
            "y": None,
            "radius": None,
            "confidence": 0.0,
            "phase": self.phase.value,
            "trajectory": [],
            "kalman_predicted": None,
            "release_point": None,
            "release_frame": None,
            "pitch_point": None,
            "pitch_frame": None,
            "speed_kmh": None,
        }

    def _maybe_auto_calibrate(self, frame: np.ndarray):
        """Attempt to auto-calibrate using known cricket ball diameter."""
        # Not yet implemented — requires knowing when ball is at known distance
        pass


# ────────────────────────────────────────────────────────────────────────
# Backward-compatible alias
# ────────────────────────────────────────────────────────────────────────

class BallTracker(FrontOnBallTracker):
    """Legacy name — delegates to FrontOnBallTracker."""
    pass
