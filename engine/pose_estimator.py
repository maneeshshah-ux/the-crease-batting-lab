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

    def process_frame(self, frame):
        """
        Process a single RGB/BGR frame and return landmarks.

        Args:
            frame: BGR image (OpenCV default)

        Returns:
            dict with:
                - landmarks: dict of {name: (x, y, z, visibility)} in normalized coords
                - landmark_list: raw mediapipe landmark list
                - raw: raw mediapipe result
                - success: bool
        """
        if frame is None:
            return {"success": False, "landmarks": {}, "landmark_list": None, "raw": None}

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.pose.process(rgb_frame)

        rgb_frame.flags.writeable = True

        if not results.pose_landmarks:
            return {"success": False, "landmarks": {}, "landmark_list": None, "raw": results}

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

        return {
            "success": True,
            "landmarks": landmarks,
            "landmark_list": landmark_list,
            "raw": results,
            "frame_height": h,
            "frame_width": w,
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
