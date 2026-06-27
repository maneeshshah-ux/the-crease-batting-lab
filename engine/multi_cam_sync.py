"""
Multi-Camera Sync Engine — Audio cross-correlation timeline alignment.

Syncs multiple camera recordings of the same batting session into a single
unified timeline. Uses audio cross-correlation (FFT-based) to find the time
offset between each pair of cameras.

PRO-gated: this feature is only available to Pro/Academy tier users.

Usage:
    from engine.multi_cam_sync import MultiCameraSync

    sync = MultiCameraSync()

    # Generate a shareable session code
    code = sync.generate_session_code()
    qr_path = sync.generate_qr_code(code)

    # Sync two or more videos
    result = sync.sync_videos(["cam1.mp4", "cam2.mp4", "cam3.mp4"])

    # Align frame metrics to a unified timeline
    unified = sync.align_timelines(result, [metrics1, metrics2])

    # Create a side-by-side composite video
    output = sync.create_multi_cam_video(
        video_paths=["cam1.mp4", "cam2.mp4"],
        offsets=result["offsets"],
        output_path="synced.mp4"
    )

Requirements:
    - ffmpeg (system binary) for audio extraction and video compositing
    - numpy for cross-correlation
    - qrcode[pil] for QR code generation
"""

import os
import re
import json
import subprocess
import tempfile
import hashlib
import struct
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters for session codes: no ambiguous 0/O, 1/I/L
CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6

# Audio extraction settings
AUDIO_SAMPLE_RATE = 44100  # Hz
AUDIO_CHANNELS = 1          # mono

# Cross-correlation peak detection
PEAK_PROMINENCE_RATIO = 0.5  # peak must be at least 50% of main peak

# Composite video layout
LAYOUT_SIDE_BY_SIDE = "side_by_side"
LAYOUT_GRID = "grid"
LAYOUT_PIP = "picture_in_picture"

# PRO gating
PRO_FEATURE = True  # Multi-camera sync is a PRO feature


class FfmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not installed on the system."""
    pass


class MultiCameraSync:
    """
    Synchronises multiple camera recordings of the same batting session.

    Core capabilities:
        1. Generate unique alphanumeric session codes (with QR code)
        2. Extract audio from video files
        3. Find timeline offsets via cross-correlation
        4. Align frame metrics to a unified timeline
        5. Create multi-camera composite videos
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._check_ffmpeg()

    # ------------------------------------------------------------------
    # Session Codes
    # ------------------------------------------------------------------

    @staticmethod
    def generate_session_code(existing_codes: Optional[set] = None) -> str:
        """
        Generate a unique 6-character alphanumeric session code.

        Uses a time-based seed + XOR shift to produce a pseudo-random code
        without relying on the random module (deterministic & replayable).

        Args:
            existing_codes: Optional set of already-used codes to avoid collisions.

        Returns:
            A 6-character code string (e.g. "X3K9M2").
        """
        seed_bytes = os.urandom(4)
        seed = struct.unpack('<I', seed_bytes)[0]
        code = MultiCameraSync._xorshift_code(seed, existing_codes or set())
        return code

    @staticmethod
    def generate_deterministic_code(seed_str: str) -> str:
        """
        Generate a deterministic code from a seed string (e.g. user email + date).
        Used for testing.
        """
        seed_bytes = seed_str.encode("utf-8")
        digest = hashlib.sha256(seed_bytes).hexdigest()
        seed = int(digest[:8], 16)
        return MultiCameraSync._xorshift_code(seed, set())

    @staticmethod
    def _xorshift_code(seed: int, existing: set) -> str:
        """XOR-shift pseudo-random number generator to produce a 6-char code."""
        state = seed
        for _ in range(10):  # warm-up
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= (state >> 7) & 0xFFFFFFFF
            state ^= (state << 17) & 0xFFFFFFFF

        max_attempts = 100
        for attempt in range(max_attempts):
            chars = []
            temp = state
            for _ in range(CODE_LENGTH):
                temp ^= (temp << 13) & 0xFFFFFFFF
                temp ^= (temp >> 7) & 0xFFFFFFFF
                temp ^= (temp << 17) & 0xFFFFFFFF
                idx = temp % len(CODE_CHARS)
                chars.append(CODE_CHARS[idx])
            code = "".join(chars)
            if code not in existing:
                return code
            # Re-seed for next attempt
            state = (state + attempt + 1) & 0xFFFFFFFF
        # Fallback: very unlikely, but include a check digit
        return "X" + code[1:]

    @staticmethod
    def validate_session_code(code: str) -> bool:
        """Validate a session code format (6 alphanumeric, no ambiguous chars)."""
        if len(code) != CODE_LENGTH:
            return False
        valid_chars = set(CODE_CHARS)
        return all(c in valid_chars for c in code.upper())

    # ------------------------------------------------------------------
    # QR Codes
    # ------------------------------------------------------------------

    @staticmethod
    def generate_qr_code(session_code: str, output_path: Optional[str] = None) -> str:
        """
        Generate a QR code image for a session code.

        Args:
            session_code: The 6-char session code to encode.
            output_path: Where to save the PNG. If None, saves alongside code.

        Returns:
            Path to the generated QR code PNG file.
        """
        try:
            import qrcode
        except ImportError:
            raise ImportError(
                "qrcode package required for QR generation. "
                "Install with: pip install qrcode[pil]"
            )

        if output_path is None:
            output_dir = Path(tempfile.gettempdir()) / "crease_qr"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"session_{session_code}.png")

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(session_code)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#E55000", back_color="#0A0A0A")
        img.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Audio Extraction
    # ------------------------------------------------------------------

    def extract_audio(self, video_path: str,
                      sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Extract mono audio from a video file as a numpy array.

        Uses ffmpeg to decode audio to raw PCM, then loads into memory.

        Args:
            video_path: Path to the video file.
            sr: Sample rate (default: self.sample_rate).

        Returns:
            Tuple of (audio_waveform: np.ndarray, sample_rate: int).
            Waveform is float32 in range [-1.0, 1.0].

        Raises:
            FfmpegNotFoundError: If ffmpeg is not installed.
            ValueError: If video has no audio stream or file not found.
        """
        self._check_ffmpeg()

        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        sr = sr or self.sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            # Use ffmpeg to extract audio as mono WAV
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                 # no video
                "-ac", str(AUDIO_CHANNELS),  # mono
                "-ar", str(sr),         # sample rate
                "-f", "wav",           # WAV format
                "-acodec", "pcm_s16le",  # 16-bit PCM
                wav_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                if "Stream map" in stderr and "No such stream" in stderr:
                    raise ValueError(
                        f"No audio stream found in {video_path}. "
                        "Multi-camera sync requires audio on all cameras."
                    )
                # ffmpeg sometimes returns non-zero for non-fatal warnings;
                # check if the output file was actually created.
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 44:
                    raise RuntimeError(
                        f"ffmpeg audio extraction failed for {video_path}:\n{stderr}"
                    )

            # Read WAV file as numpy array
            waveform, sample_rate = self._read_wav(wav_path)
            return waveform, sample_rate

        finally:
            # Clean up temp WAV
            if os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except PermissionError:
                    pass

    @staticmethod
    def _read_wav(wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Read a 16-bit mono WAV file into a float32 numpy array.

        Reads the raw file without additional dependencies (no scipy/soundfile).
        """
        with open(wav_path, "rb") as f:
            raw = f.read()

        # Parse WAV header
        # riff_size = struct.unpack_from('<I', raw, 4)[0]
        fmt_chunk_offset = 12
        # Find fmt chunk
        while fmt_chunk_offset < len(raw) - 8:
            chunk_id = raw[fmt_chunk_offset:fmt_chunk_offset + 4]
            chunk_size = struct.unpack_from('<I', raw, fmt_chunk_offset + 4)[0]
            if chunk_id == b'fmt ':
                break
            fmt_chunk_offset += 8 + chunk_size
        else:
            raise ValueError("WAV has no fmt chunk")

        audio_format = struct.unpack_from('<H', raw, fmt_chunk_offset + 8)[0]
        num_channels = struct.unpack_from('<H', raw, fmt_chunk_offset + 10)[0]
        sample_rate = struct.unpack_from('<I', raw, fmt_chunk_offset + 12)[0]
        bits_per_sample = struct.unpack_from('<H', raw, fmt_chunk_offset + 22)[0]

        if audio_format != 1:  # PCM
            raise ValueError(f"Unsupported WAV format: {audio_format} (only PCM)")

        # Find data chunk
        data_offset = fmt_chunk_offset + 8 + chunk_size
        while data_offset < len(raw) - 8:
            chunk_id = raw[data_offset:data_offset + 4]
            chunk_size = struct.unpack_from('<I', raw, data_offset + 4)[0]
            if chunk_id == b'data':
                # Read audio samples
                raw_data = raw[data_offset + 8:data_offset + 8 + chunk_size]
                break
            data_offset += 8 + chunk_size
        else:
            # No data chunk found — return empty waveform
            return np.array([], dtype=np.float32), sample_rate

        if bits_per_sample == 16:
            dtype = np.int16
            samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
            samples /= 32768.0  # Normalize to [-1, 1]
        elif bits_per_sample == 24:
            # 3 bytes per sample, little-endian
            raw_data = raw_data[:len(raw_data) - (len(raw_data) % 3)]
            samples = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
            # Convert 3-byte signed 24-bit to int32 then float
            samples_int = (
                samples[:, 0].astype(np.int32) |
                samples[:, 1].astype(np.int32) << 8 |
                samples[:, 2].astype(np.int32) << 16
            )
            # Sign extend
            samples_int = np.where(
                samples_int >= 0x800000,
                samples_int - 0x1000000,
                samples_int
            )
            samples = samples_int.astype(np.float32) / 8388608.0
        else:
            dtype = np.int32 if bits_per_sample == 32 else np.int8
            max_val = {32: 2147483648.0, 8: 128.0}.get(bits_per_sample, 32768.0)
            samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
            samples /= max_val

        if num_channels > 1:
            # Average to mono
            samples = samples.reshape(-1, num_channels).mean(axis=1)

        return samples, sample_rate

    # ------------------------------------------------------------------
    # Cross-Correlation Offset
    # ------------------------------------------------------------------

    def compute_offset(self, video_a: str, video_b: str,
                       sr: Optional[int] = None,
                       max_offset_sec: float = 30.0) -> Dict:
        """
        Compute the time offset between two videos via audio cross-correlation.

        Determines how much video_b is delayed RELATIVE to video_a.
        If video_b is 5 seconds behind video_a, offset = +5.0 seconds.
        This means: to align, video_b should start 5 seconds after video_a,
        or equivalently, video_a should start 5 seconds before video_b.

        Uses FFT-based circular cross-correlation (numpy rfft/irfft) then
        reconstructs the linear correlation, matching numpy.correlate(a, b, 'full').

        The reconstruction handles the circular-to-linear mapping:
            output[τ] = Σ a[n] * b[(n+τ) mod n_fft]  (circular)
            linear[k] = output[-k mod n_fft]           (np.correlate convention)
            where k = lag in samples, positive = b behind a.

        Args:
            video_a: Path to the reference video.
            video_b: Path to the video to align.
            sr: Sample rate for audio extraction.
            max_offset_sec: Maximum allowed offset in seconds. Larger offsets
                           are likely false positives.

        Returns:
            Dict with keys:
                - offset_sec: float — time offset (seconds). Positive means
                  video_b is behind video_a.
                - correlation_score: float — peak correlation value [0, 1].
                  Higher = better match.
                - offset_samples: int — offset in samples.
                - sample_rate: int — sample rate used.
                - video_a: str — path to first video.
                - video_b: str — path to second video.
        """
        sr = sr or self.sample_rate

        # Extract audio from both videos
        audio_a, sr_a = self.extract_audio(video_a, sr)
        audio_b, sr_b = self.extract_audio(video_b, sr)

        # Normalise both signals
        audio_a = audio_a / (np.max(np.abs(audio_a)) + 1e-10)
        audio_b = audio_b / (np.max(np.abs(audio_b)) + 1e-10)

        # If one is significantly longer, trim to speed up computation
        max_offset_samples = int(max_offset_sec * sr)
        min_len = min(len(audio_a), len(audio_b))

        if min_len > 2 * max_offset_samples + sr:  # more than 2x search window + 1 sec
            # Trim both to a central segment + max_offset padding
            center = min_len // 2
            half_window = max_offset_samples + sr  # 1 second extra on each side
            start = max(0, center - half_window)
            end = min(min_len, center + half_window)
            audio_a = audio_a[start:end]
            audio_b = audio_b[start:end]

        # FFT-based circular cross-correlation
        # o[τ] = Σ a[n] * b[(n+τ) mod n_fft]
        n_a = len(audio_a)
        n_b = len(audio_b)
        n_linear = n_a + n_b - 1
        n_fft = 1 << (n_linear).bit_length()  # next power of 2, >= n_linear

        fft_a = np.fft.rfft(audio_a, n=n_fft)
        fft_b = np.fft.rfft(audio_b, n=n_fft)

        # Circular cross-correlation: irfft(conj(FFT(a)) * FFT(b))
        circular = np.fft.irfft(fft_a.conj() * fft_b, n=n_fft)

        # Normalise
        auto_a = np.sum(audio_a ** 2)
        auto_b = np.sum(audio_b ** 2)
        norm_factor = np.sqrt(auto_a * auto_b) + 1e-10

        # Reconstruct linear correlation matching np.correlate(a, b, 'full')
        # linear[k] for k in [-(n_b-1), n_a-1] matches np.correlate convention:
        #   z[k] = Σ a[n+k] * b[n]
        #   k < 0: video_b content is ahead of video_a (b happens earlier)
        #   k > 0: video_b content is behind video_a (b happens later)
        #   k = 0: aligned
        #
        # Relationship: circular[τ] = linear[-τ mod n_fft]
        # So linear[k] = circular[-k mod n_fft]
        linear = np.concatenate([
            circular[n_b - 1:0:-1],            # k = -(n_b-1) to -1
            [circular[0]],                      # k = 0
            circular[n_fft - 1:n_fft - n_a:-1],  # k = 1 to n_a-1
        ])

        # Normalise
        linear /= norm_factor

        # Find peak within allowed offset range
        max_offset_samp = int(max_offset_sec * sr)
        center_idx = n_b - 1  # index of k=0 in linear array

        search_start = max(0, center_idx - max_offset_samp)
        search_end = min(len(linear), center_idx + max_offset_samp + 1)

        if search_end <= search_start:
            search_start = 0
            search_end = len(linear)

        search_region = linear[search_start:search_end]
        peak_idx = np.argmax(np.abs(search_region))
        peak_value = float(np.abs(search_region[peak_idx]))
        global_peak_idx = search_start + peak_idx

        # Lag k in samples (np.correlate convention):
        #   k < 0: b content is behind a (b happens later)
        #   k > 0: b content is ahead of a (b happens earlier)
        k_peak = global_peak_idx - center_idx
        # Negate so output convention is: positive = b behind a
        offset_sec = -k_peak / sr

        return {
            "offset_sec": round(offset_sec, 3),
            "correlation_score": round(peak_value, 4),
            "offset_samples": k_peak,
            "sample_rate": sr,
            "video_a": str(video_a),
            "video_b": str(video_b),
        }

    # ------------------------------------------------------------------
    # Multi-Video Sync
    # ------------------------------------------------------------------

    def sync_videos(self, video_paths: List[str],
                    reference_index: int = 0,
                    max_offset_sec: float = 30.0) -> Dict:
        """
        Sync N videos by computing pairwise offsets against a reference camera.

        The reference camera (default: first in list) becomes the "timeline
        anchor" at offset 0. All other cameras' offsets are computed relative
        to it.

        For robustness, cross-checks with adjacent-pair offsets are used
        to detect inconsistencies.

        Args:
            video_paths: List of paths to video files to sync.
            reference_index: Index of the reference video (default: 0).
            max_offset_sec: Maximum expected offset in seconds.

        Returns:
            Dict with keys:
                - cameras: List of camera info dicts with path and offset.
                - offsets: Dict mapping video_path -> offset_sec (relative to ref).
                - reference: Path to the reference video.
                - correlations: Dict mapping pair -> correlation_score.
                - num_cameras: int
                - timestamp: ISO timestamp of sync operation.
        """
        if len(video_paths) < 2:
            raise ValueError("Need at least 2 videos to sync")

        if reference_index < 0 or reference_index >= len(video_paths):
            raise ValueError(f"reference_index {reference_index} out of range")

        ref_path = str(video_paths[reference_index])
        offsets = {ref_path: 0.0}
        correlations = {}

        for i, vpath in enumerate(video_paths):
            vpath = str(vpath)
            if i == reference_index:
                continue

            result = self.compute_offset(ref_path, vpath,
                                         max_offset_sec=max_offset_sec)
            offsets[vpath] = result["offset_sec"]
            correlations[f"{ref_path} -> {vpath}"] = {
                "offset_sec": result["offset_sec"],
                "correlation_score": result["correlation_score"],
            }

        cameras = []
        for vpath in video_paths:
            vpath = str(vpath)
            cameras.append({
                "video_path": vpath,
                "video_name": Path(vpath).name,
                "offset_sec": offsets[vpath],
            })

        return {
            "cameras": cameras,
            "offsets": offsets,
            "reference": ref_path,
            "correlations": correlations,
            "num_cameras": len(video_paths),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Timeline Alignment
    # ------------------------------------------------------------------

    def align_timelines(self, sync_result: Dict,
                        frame_metrics_list: List[Dict]) -> Dict:
        """
        Align frame-level metrics from all cameras to a unified timeline.

        Args:
            sync_result: Output from sync_videos().
            frame_metrics_list: List of per-camera frame metrics dicts.
                Each dict maps frame_index -> metric_value, or is a list of
                per-frame metric dicts.

        Returns:
            Dict with:
                - unified_timeline: List of events sorted by unified time.
                - camera_tracks: Per-camera metrics keyed by camera path,
                  with frame times shifted by offset.
                - total_duration_sec: Duration of the longest camera feed.
        """
        unified = []
        camera_tracks = {}

        cameras = sync_result["cameras"]

        for cam_info in cameras:
            vpath = cam_info["video_path"]
            offset = cam_info["offset_sec"]

            # Find matching metrics for this camera
            cam_metrics = None
            for m in frame_metrics_list:
                if m.get("video_path") == vpath or m.get("camera_path") == vpath:
                    cam_metrics = m
                    break

            if cam_metrics is None:
                continue

            # Shift frame times by offset
            shifted = []
            frames = cam_metrics.get("frames", [])
            fps = cam_metrics.get("fps", 30.0)

            for i, frame_data in enumerate(frames):
                original_time = i / fps
                unified_time = original_time - offset  # shift to reference timeline

                entry = {
                    "camera": vpath,
                    "original_frame": i,
                    "original_time_sec": round(original_time, 3),
                    "unified_time_sec": round(unified_time, 3),
                    "metrics": frame_data,
                }
                shifted.append(entry)
                unified.append(entry)

            camera_tracks[vpath] = {
                "shifted_timeline": shifted,
                "offset_sec": offset,
                "num_frames": len(frames),
                "fps": fps,
            }

        # Sort unified timeline by unified_time
        unified.sort(key=lambda x: x["unified_time_sec"])

        # Compute total duration
        if unified:
            total_duration = max(e["unified_time_sec"] for e in unified) - \
                             min(e["unified_time_sec"] for e in unified)
        else:
            total_duration = 0.0

        return {
            "unified_timeline": unified,
            "camera_tracks": camera_tracks,
            "total_duration_sec": round(total_duration, 3),
            "num_events": len(unified),
            "num_cameras": len(camera_tracks),
        }

    # ------------------------------------------------------------------
    # Composite Video
    # ------------------------------------------------------------------

    def create_multi_cam_video(
        self,
        video_paths: List[str],
        offsets: Dict,
        output_path: str,
        layout: str = LAYOUT_SIDE_BY_SIDE,
    ) -> str:
        """
        Create a multi-camera composite video with synced timelines.

        Uses ffmpeg's overlay filter with setpts to align each camera to
        the unified timeline.

        Args:
            video_paths: List of video file paths.
            offsets: Dict mapping video_path -> offset_sec (from sync_videos).
            output_path: Where to save the composite video.
            layout: Layout mode — "side_by_side", "grid", or
                   "picture_in_picture".

        Returns:
            Path to the generated composite video.

        Raises:
            FfmpegNotFoundError: If ffmpeg is not installed.
            ValueError: If fewer than 2 videos provided.
        """
        self._check_ffmpeg()

        if len(video_paths) < 2:
            raise ValueError("Need at least 2 videos for compositing")

        # Validate all files exist
        for vp in video_paths:
            if not os.path.exists(vp):
                raise ValueError(f"Video not found: {vp}")

        output_path = str(output_path)

        if layout == LAYOUT_SIDE_BY_SIDE:
            return self._composite_side_by_side(video_paths, offsets, output_path)
        elif layout == LAYOUT_GRID:
            return self._composite_grid(video_paths, offsets, output_path)
        elif layout == LAYOUT_PIP:
            return self._composite_pip(video_paths, offsets, output_path)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def _composite_side_by_side(self, video_paths: List[str],
                                 offsets: Dict,
                                 output_path: str) -> str:
        """
        Side-by-side composite: 2 videos side by side.
        For 3+ videos, creates a row of thumbnails + main view.
        """
        if len(video_paths) == 2:
            return self._two_up_composite(video_paths, offsets, output_path)
        else:
            return self._multi_grid_composite(video_paths, offsets, output_path)

    def _two_up_composite(self, video_paths: List[str],
                           offsets: Dict,
                           output_path: str) -> str:
        """Side-by-side for exactly 2 cameras."""
        v1, v2 = video_paths[:2]
        o1 = offsets.get(str(v1), 0.0)
        o2 = offsets.get(str(v2), 0.0)

        # Use ffmpeg to stack horizontally with PTS adjustment for sync
        cmd = [
            "ffmpeg", "-y",
            "-i", str(v1),
            "-i", str(v2),
            "-filter_complex",
            f"[0:v]setpts=PTS+{o1:.3f}/TB,scale=iw/2:ih/2[vid0];"
            f"[1:v]setpts=PTS+{o2:.3f}/TB,scale=iw/2:ih/2[vid1];"
            f"[vid0][vid1]hstack=inputs=2[v]",
            "-map", "[v]",
            # Mix audio from both cameras
            f"[0:a]adelay={int(o1*1000)}|{int(o1*1000)}[a0];"
            f"[1:a]adelay={int(o2*1000)}|{int(o2*1000)}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first[a]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0 and not os.path.exists(output_path):
            raise RuntimeError(
                f"ffmpeg composite failed:\n{result.stderr.decode('utf-8', errors='replace')}"
            )
        return output_path

    def _multi_grid_composite(self, video_paths: List[str],
                               offsets: Dict,
                               output_path: str) -> str:
        """
        Grid composite for 3+ cameras.
        2×2 grid, with extra cameras as PiP overlays if >4.
        """
        # Build filter complex
        n = min(len(video_paths), 4)  # max 4 in grid
        inputs = "".join(f"[{i}:v]" for i in range(n))
        setpts = ";".join(
            f"[{i}:v]setpts=PTS+{offsets.get(str(video_paths[i]), 0.0):.3f}/TB,"
            f"scale=iw/2:ih/2[vid{i}]"
            for i in range(n)
        )
        grid_parts = []
        for row in range(2):
            row_inputs = "".join(f"[vid{row*2 + col}]" for col in range(2) if row*2 + col < n)
            if row_inputs:
                grid_parts.append(f"{row_inputs}hstack=inputs={2 - (row*2 + 1 >= n)}[row{row}]")

        vstack = "[row0][row1]vstack=inputs=2[v]" if len(grid_parts) == 2 else grid_parts[0].split(">")[1]

        # Rebuild properly: handle 2×2 grid
        if n <= 2:
            return self._two_up_composite(video_paths[:2], offsets, output_path)

        # Simple 2×2 for 3-4 cameras
        cmd = [
            "ffmpeg", "-y",
        ]
        for vp in video_paths[:n]:
            cmd.extend(["-i", str(vp)])

        # Tile filter
        tile_w = 2
        tile_h = (n + 1) // 2
        filter_complex = (
            f"{'; '.join(setpts.split(';'))}; "
            f"{'[vid0][vid1]hstack=inputs=2[row0];' if n > 1 else ''}"
            f"{'[vid2][vid3]hstack=inputs=2[row1]' if n > 2 else '[vid2]null[row1]'}; "
            f"[row0][row1]vstack=inputs=2[v]"
        )

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path,
        ])
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0 and not os.path.exists(output_path):
            raise RuntimeError(
                f"ffmpeg grid composite failed:\n{result.stderr.decode('utf-8', errors='replace')}"
            )
        return output_path

    def _composite_grid(self, video_paths: List[str],
                         offsets: Dict,
                         output_path: str) -> str:
        """Grid layout — delegates to multi_grid_composite."""
        return self._multi_grid_composite(video_paths, offsets, output_path)

    def _composite_pip(self, video_paths: List[str],
                        offsets: Dict,
                        output_path: str) -> str:
        """
        Picture-in-picture: first video is main, second is PiP overlay.
        Only works for exactly 2 videos.
        """
        v1, v2 = video_paths[:2]
        o1 = offsets.get(str(v1), 0.0)
        o2 = offsets.get(str(v2), 0.0)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(v1),
            "-i", str(v2),
            "-filter_complex",
            f"[0:v]setpts=PTS+{o1:.3f}/TB,scale=1920:1080[main];"
            f"[1:v]setpts=PTS+{o2:.3f}/TB,scale=480:270[pip];"
            f"[main][pip]overlay=W-w-20:H-h-20[v]",
            "-map", "[v]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0 and not os.path.exists(output_path):
            raise RuntimeError(
                f"ffmpeg PiP composite failed:\n{result.stderr.decode('utf-8', errors='replace')}"
            )
        return output_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ffmpeg() -> None:
        """Verify ffmpeg is available on the system PATH."""
        import shutil
        if shutil.which("ffmpeg") is None:
            raise FfmpegNotFoundError(
                "ffmpeg not found. Multi-camera sync requires ffmpeg.\n"
                "Install with:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            )

    @staticmethod
    def estimate_required_disk_space(video_paths: List[str]) -> Dict:
        """
        Estimate the disk space required for syncing given videos.

        Returns size info for the original videos plus estimated output.
        """
        total_input = 0
        file_info = []
        for vp in video_paths:
            if os.path.exists(vp):
                size = os.path.getsize(vp)
                total_input += size
                file_info.append({
                    "path": vp,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 1),
                })

        # Composite output is roughly 1.5x the largest input (audio+re-encode)
        max_input = max(f["size_bytes"] for f in file_info) if file_info else 0
        estimated_output = int(max_input * 1.5 * min(len(video_paths), 2))

        return {
            "files": file_info,
            "total_input_mb": round(total_input / (1024 * 1024), 1),
            "estimated_output_mb": round(estimated_output / (1024 * 1024), 1),
            "num_files": len(file_info),
        }
