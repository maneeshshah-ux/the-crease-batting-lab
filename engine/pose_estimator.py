"""
Pose Estimator — MediaPipe-based full-body pose detection for cricket batting.

Extracts 33 body landmarks per frame with optional handedness tracking.
Designed for side-on batting analysis (classic coaching view).
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import urllib.request

# ── Read-only filesystem workaround ─────────────────────────────────────
# MediaPipe 0.10.9 bundles most model files in the pip wheel, but the
# model_complexity-specific pose landmark models (lite & heavy) are NOT
# bundled — they are downloaded at runtime via download_oss_model().
#
# On Render.com (read-only filesystem) the download fails with
# PermissionError because site-packages is not writable.
#
# Our fix:
#   1. Pre-download the missing .tflite files to /tmp/
#   2. Copy bundled .tflite files that would be masked by a resource-dir
#      change to /tmp/ as well
#   3. Redirect MediaPipe's C++ resource directory (resource_util) to /tmp/
#      AFTER SolutionBase.__init__ sets it (because SolutionBase.__init__
#      always resets it to site-packages).
#
# The C++ calculators (PoseDetection, PoseLandmarkByRoiCpu) load model
# files relative to resource_util's resource dir.  By pointing it at /tmp/
# they find the files we pre-downloaded.
# ──────────────────────────────────────────────────────────────────────────

# Models NOT bundled in the wheel (downloaded at runtime)
_MP_MODEL_DOWNLOAD_RELPATHS = {
    0: "mediapipe/modules/pose_landmark/pose_landmark_lite.tflite",
    2: "mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite",
}

# Models bundled in the wheel but needed after resource-dir redirect
# (must be copied to /tmp/ so they remain findable)
_MP_MODEL_COPY_RELPATHS = [
    "mediapipe/modules/pose_detection/pose_detection.tflite",
    "mediapipe/modules/pose_landmark/pose_landmark_full.tflite",
]


def _ensure_pose_models(model_complexity: int = 0):
    """Pre-download/copy needed model files and redirect resource dir.

    Must be called AFTER ``SolutionBase.__init__`` (i.e. after creating the
    ``Pose`` object) so that our ``resource_util.set_resource_dir()`` call
    is not overwritten.
    """
    rel = _MP_MODEL_DOWNLOAD_RELPATHS.get(model_complexity)

    try:
        from mediapipe.python.solutions import download_utils as _du
        from mediapipe.python._framework_bindings import resource_util as _ru
    except ImportError:
        return

    mp_root = os.sep.join(
        os.path.abspath(_du.__file__).split(os.sep)[:-4]
    )

    # ── Only act if site-packages is read-only ──────────────────────────
    if os.access(mp_root, os.W_OK):
        return

    # ── 1. Download missing pose landmark model to /tmp/ ────────────────
    if rel is not None:
        tmp_model = os.path.join("/tmp", rel)
        os.makedirs(os.path.dirname(tmp_model), exist_ok=True)
        if not os.path.exists(tmp_model):
            model_url = _du._GCS_URL_PREFIX + rel.split("/")[-1]
            with urllib.request.urlopen(model_url) as resp, \
                    open(tmp_model, "wb") as f:
                shutil.copyfileobj(resp, f)

    # ── 2. Copy bundled models that will be masked by the dir redirect ──
    for copy_rel in _MP_MODEL_COPY_RELPATHS:
        tmp_path = os.path.join("/tmp", copy_rel)
        if os.path.exists(tmp_path):
            continue
        src = os.path.join(mp_root, copy_rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            shutil.copy2(src, tmp_path)

    # ── 3. Redirect resource dir so C++ calculators find models in /tmp/ ─
    _ru.set_resource_dir("/tmp/")

    # ── 4. Patch download_oss_model as safety net for any lazy downloads ─
    _original_download = _du.download_oss_model

    def _patched_download(model_path: str):
        """Download model to /tmp/ when site-packages is read-only."""
        full_path = os.path.join(mp_root, model_path)
        if os.path.exists(full_path):
            return
        target_dir = os.path.dirname(full_path)
        if os.access(target_dir, os.W_OK):
            _original_download(model_path)
            return
        local_path = os.path.join("/tmp", model_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            return
        url = _du._GCS_URL_PREFIX + model_path.split("/")[-1]
        with urllib.request.urlopen(url) as resp, \
                open(local_path, "wb") as f:
            shutil.copyfileobj(resp, f)

    _du.download_oss_model = _patched_download

# ── MediaPipe API compatibility ──────────────────────────────────────────
# Different MediaPipe versions use different import paths.
# 0.10.x uses mp.solutions.*, newer versions may use mp.python.solutions.*
try:
    _mp_pose = mp.solutions.pose
    _mp_drawing = mp.solutions.drawing_utils
    _mp_has_solutions = True
except AttributeError:
    # Fallback: try the new-style API
    try:
        _mp_pose = mp.tasks.vision.PoseLandmarker
        _mp_drawing = mp.tasks.vision
        _mp_has_solutions = False
    except AttributeError:
        # Last resort: try mediapipe.python.solutions
        try:
            import mediapipe.python.solutions as mp_sol
            _mp_pose = mp_sol.pose
            _mp_drawing = mp_sol.drawing_utils
            _mp_has_solutions = True
        except (ImportError, AttributeError):
            raise ImportError(
                "Could not find MediaPipe Pose API. "
                "Try: pip install mediapipe==0.10.9"
            )

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

    def __init__(self, static_mode=False, model_complexity=0, smooth=True,
                 min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Args:
            static_mode: False for video, True for single images
            model_complexity: 0=lite, 1=full, 2=heavy
            smooth: Temporal smoothing across frames
        """
        # Use module-level compatibility references
        if _mp_has_solutions:
            self.pose = _mp_pose.Pose(
                static_image_mode=static_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            # Ensure pose model is available on read-only filesystems.
            # Must happen AFTER Pose.__init__ (which calls
            # SolutionBase.__init__ and resets resource_util).
            _ensure_pose_models(model_complexity)
        else:
            # New-style MediaPipe API (pose_landmarker based)
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe import model_ckpt_util
            # Use default model
            model_path = model_ckpt_util.get_model('pose_landmarker_lite')
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode='video',
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
            )
            self.pose = PoseLandmarker.create_from_options(options)

        self.mp_drawing = _mp_drawing
        self.mp_pose = _mp_pose

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
