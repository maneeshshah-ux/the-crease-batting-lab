"""
Ball Tracker — Multi-strategy ball detection and trajectory tracking.

Uses:
- Colour-based detection (red/pink/white ball)
- Motion-based detection (frame differencing)
- Kalman filtering for trajectory smoothing
- Optional: simple background subtraction
"""

import cv2
import numpy as np
from collections import deque


class KalmanTracker:
    """Simple Kalman filter for 2D ball position smoothing."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                 [0, 1]], np.float32) * 0.5
        self.initialized = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return (int(pred[0]), int(pred[1]))

    def update(self, x, y):
        if not self.initialized:
            self.init(x, y)
            return (x, y)
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measured)
        return (int(corrected[0]), int(corrected[1]))


class BallTracker:
    """
    Multi-strategy ball tracker for cricket nets/grounds.

    Strategies (tried in order):
    1. Colour-based: Red ball (cricket red) or white ball in HSV space
    2. Motion-based: Frame differencing + circularity detection
    3. Combined: Colour mask + motion mask intersection
    """

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
    }

    def __init__(self, ball_color="red", min_radius=3, max_radius=30,
                 use_kalman=True, history_frames=30):
        """
        Args:
            ball_color: 'red', 'pink', or 'white'
            min_radius: minimum ball radius in pixels
            max_radius: maximum ball radius in pixels
            use_kalman: apply Kalman filtering
            history_frames: frames to keep for trajectory
        """
        self.ball_color = ball_color
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.use_kalman = use_kalman

        self.kalman = KalmanTracker() if use_kalman else None
        self.trajectory = deque(maxlen=history_frames)
        self.prev_gray = None
        self.ball_positions = []  # list of (frame_idx, x, y)
        self.detection_confidences = []

    def detect_color_ball(self, frame):
        """Detect ball by colour in HSV space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        ranges = self.BALL_COLOR_RANGES.get(self.ball_color, self.BALL_COLOR_RANGES["red"])

        if self.ball_color == "red":
            # Red ball wraps around hue 0/180
            mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # too small
                continue

            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.4:  # not circular enough
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            if radius < self.min_radius or radius > self.max_radius:
                continue

            # Score based on circularity and size
            score = circularity * min(radius / 15.0, 2.0)

            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(radius))

        return best, best_score

    def detect_motion_ball(self, frame):
        """Detect moving ball via frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0

        # Frame differencing
        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Find moving regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            score = min(area / 100.0, 2.0)
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(radius))

        self.prev_gray = gray
        return best, best_score

    def track(self, frame, frame_idx=0):
        """
        Track ball in the given frame.

        Returns:
            dict with ball position, confidence, trajectory
        """
        result = {
            "detected": False,
            "x": None,
            "y": None,
            "radius": None,
            "confidence": 0.0,
            "trajectory": [],
            "kalman_predicted": None,
        }

        # Strategy 1: Colour detection (primary)
        color_ball, color_conf = self.detect_color_ball(frame)

        # Strategy 2: Motion detection (secondary)
        motion_ball, motion_conf = self.detect_motion_ball(frame)

        # Combine strategies
        best_pos = None
        best_conf = 0
        best_radius = None

        if color_ball and color_conf > best_conf:
            best_pos = (color_ball[0], color_ball[1])
            best_conf = color_conf
            best_radius = color_ball[2]

        if motion_ball and motion_conf > best_conf:
            best_pos = (motion_ball[0], motion_ball[1])
            best_conf = motion_conf
            best_radius = motion_ball[2]

        # If both agree (within 20px), boost confidence
        if color_ball and motion_ball:
            dist = np.sqrt((color_ball[0] - motion_ball[0])**2 +
                           (color_ball[1] - motion_ball[1])**2)
            if dist < 20:
                best_conf = max(best_conf, (color_conf + motion_conf) * 0.8)

        if best_pos and best_conf > 0.3:
            # Apply Kalman filter
            if self.use_kalman and self.kalman:
                kalman_pred = self.kalman.predict()
                result["kalman_predicted"] = kalman_pred

                smoothed = self.kalman.update(best_pos[0], best_pos[1])
                bx, by = smoothed
            else:
                bx, by = best_pos

            result["detected"] = True
            result["x"] = bx
            result["y"] = by
            result["radius"] = best_radius
            result["confidence"] = best_conf

            self.ball_positions.append((frame_idx, bx, by))
            self.trajectory.append((bx, by))
            self.detection_confidences.append(best_conf)

        result["trajectory"] = list(self.trajectory)
        return result

    def get_smoothed_trajectory(self, window=5):
        """Get smoothed ball trajectory."""
        if len(self.ball_positions) < window:
            return self.ball_positions

        smoothed = []
        positions = np.array([(p[1], p[2]) for p in self.ball_positions])

        for i in range(len(positions)):
            start = max(0, i - window // 2)
            end = min(len(positions), i + window // 2 + 1)
            avg = positions[start:end].mean(axis=0)
            smoothed.append((self.ball_positions[i][0], int(avg[0]), int(avg[1])))

        return smoothed

    def estimate_speed(self, fps, calibration_mm_per_pixel=None):
        """
        Estimate ball speed from trajectory.
        If calibration is known, returns km/h.
        Otherwise returns pixels/frame velocity.

        Args:
            fps: frames per second of video
            calibration_mm_per_pixel: physical size per pixel

        Returns:
            dict with speed metrics
        """
        if len(self.ball_positions) < 5:
            return {"estimated": False, "speed_px_per_frame": 0}

        positions = np.array([(p[1], p[2]) for p in self.ball_positions])
        velocities = np.diff(positions, axis=0)
        speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)

        avg_speed_px = np.mean(speeds)

        result = {
            "estimated": True,
            "speed_px_per_frame": float(avg_speed_px),
            "speed_px_per_sec": float(avg_speed_px * fps),
        }

        if calibration_mm_per_pixel:
            speed_mm_per_sec = avg_speed_px * fps * calibration_mm_per_pixel
            speed_kmh = speed_mm_per_sec * 3.6 / 1e6
            result["speed_kmh"] = float(speed_kmh)
            result["speed_mph"] = float(speed_kmh * 0.621371)

        return result

    def reset(self):
        """Reset tracker state for new session."""
        self.trajectory.clear()
        self.ball_positions = []
        self.detection_confidences = []
        self.prev_gray = None
        if self.kalman:
            self.kalman = KalmanTracker()
