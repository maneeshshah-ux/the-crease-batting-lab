"""
Pose Estimator — MediaPipe-based full-body pose detection for cricket batting.

Extracts 33 body landmarks per frame with optional handedness tracking.
Designed for side-on batting analysis (classic coaching view).
"""

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe landmark indices relevant to batting analysis
BATTING_LANDMARKS = {
    # Lower body
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    # Torso
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    # Head
    "NOSE": 0,
    "LEFT_EYE": 2,
    "RIGHT_EYE": 5,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    # Hands (MediaPipe Pose gives wrist, for fingers we'd need Hands model)
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    # Heel/foot
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# Symmetry pairs for cricket stance analysis
BATTING_SYMMETRY_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_KNEE", "RIGHT_KNEE"),
    ("LEFT_ANKLE", "RIGHT_ANKLE"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
]


class PoseEstimator:
    """MediaPipe Pose wrapper for batting analysis."""

    def __init__(self, static_mode=False, model_complexity=1, smooth=True,
                 min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Args:
            static_mode: False for video, True for single images
            model_complexity: 0=lite, 1=full, 2=heavy
            smooth: Temporal smoothing across frames
        """
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Drawing utilities (mediapipe 0.10.x uses drawing_utils)
        try:
            self.mp_drawing = mp.solutions.drawing_utils
        except AttributeError:
            self.mp_drawing = mp.solutions.pose_utils
        self.mp_pose = mp.solutions.pose

    def process_frame(self, frame, prefer_batter=True):
        """
        Process a single RGB/BGR frame and return landmarks.

        Args:
            frame: BGR image (OpenCV default)
            prefer_batter: If True, filter detections that look like
                           wicketkeeper (too high in frame, too small)

        Returns:
            dict with:
                - landmarks: dict of {name: (x, y, z, visibility)} in normalized coords
                - landmark_list: raw mediapipe landmark list
                - raw: raw mediapipe result
                - success: bool
                - is_batter: bool (True if detected person appears to be the batter)
        """
        if frame is None:
            return {"success": False, "landmarks": {}, "landmark_list": None,
                    "raw": None, "is_batter": False}

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.pose.process(rgb_frame)

        rgb_frame.flags.writeable = True

        if not results.pose_landmarks:
            return {"success": False, "landmarks": {}, "landmark_list": None,
                    "raw": results, "is_batter": False}

        h, w = frame.shape[:2]
        landmarks = {}
        landmark_list = results.pose_landmarks.landmark

        for name, idx in BATTING_LANDMARKS.items():
            lm = landmark_list[idx]
            landmarks[name] = {
                "x": lm.x,  # normalized [0..1]
                "y": lm.y,  # normalized [0..1]
                "z": lm.z,  # depth (meters, roughly)
                "visibility": lm.visibility,
                "pixel_x": int(lm.x * w),
                "pixel_y": int(lm.y * h),
            }

        # Heuristic: check if this looks like a batter (vs wicketkeeper)
        is_batter = True
        if prefer_batter:
            # Batter should have feet near bottom of frame (y > 0.6)
            # Wicketkeeper stands further back → appears higher (smaller y)
            foot_y = None
            for foot_key in ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
                             "LEFT_ANKLE", "RIGHT_ANKLE"):
                if foot_key in landmarks and landmarks[foot_key]["visibility"] > 0.5:
                    foot_y = landmarks[foot_key]["y"]
                    break
            if foot_y is not None and foot_y < 0.5:
                # Feet high in frame — likely wicketkeeper or partial detection
                is_batter = False

            # Additional heuristic: if body is high in frame and looks small
            # (e.g. visible nose y < 0.3), it's likely a wicketkeeper in the background
            if is_batter and "NOSE" in landmarks and landmarks["NOSE"]["visibility"] > 0.5:
                nose_y = landmarks["NOSE"]["y"]
                if nose_y < 0.15 and foot_y is None:
                    # Face very high in frame, no feet visible — probably a distant figure
                    is_batter = False

        return {
            "success": True,
            "landmarks": landmarks,
            "landmark_list": landmark_list,
            "raw": results,
            "frame_height": h,
            "frame_width": w,
            "is_batter": is_batter,
        }

    def get_landmark_array(self, landmarks_dict):
        """
        Convert landmarks dict to numpy array for math operations.
        Returns (33, 3) array of [x, y, z] normalized coordinates.
        """
        arr = np.zeros((33, 3), dtype=np.float32)
        for name, idx in BATTING_LANDMARKS.items():
            if name in landmarks_dict:
                arr[idx] = [landmarks_dict[name]["x"],
                            landmarks_dict[name]["y"],
                            landmarks_dict[name]["z"]]
        return arr

    def draw_landmarks(self, frame, pose_result, draw_connections=True,
                       landmark_color=(0, 255, 0), connection_color=(255, 0, 0)):
        """Draw pose landmarks on a frame."""
        if pose_result["raw"] and pose_result["raw"].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_result["raw"].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=landmark_color, thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=connection_color, thickness=2),
            )
        return frame

    def close(self):
        self.pose.close()
