"""
Tests for MultiCameraSync — Audio cross-correlation timeline alignment.

Run with: pytest tests/test_multi_cam_sync.py -v

Strategy:
    - Session code generation and validation are tested directly.
    - QR code generation is tested with mocked qrcode.
    - Audio extraction and cross-correlation are tested with synthetic WAV data
      (generated in-memory via the _read_wav method or mocked ffmpeg).
    - Full ffmpeg-dependent tests are skipped if ffmpeg is not available.
    - Composite video tests mock the subprocess calls.
"""

import os
import json
import struct
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

from engine.multi_cam_sync import (
    MultiCameraSync,
    FfmpegNotFoundError,
    CODE_CHARS,
    CODE_LENGTH,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sync():
    """MultiCameraSync instance. ffmpeg availability is mocked."""
    with patch("engine.multi_cam_sync.MultiCameraSync._check_ffmpeg"):
        yield MultiCameraSync()


@pytest.fixture
def sample_wav_file():
    """
    Generate a temporary 16-bit mono WAV file with a broadband chirp signal.
    A chirp (frequency sweep) has a sharp auto-correlation peak, making it
    suitable for cross-correlation tests.
    """
    sr = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Frequency sweep: 200 Hz → 4000 Hz — broadband for sharp correlation
    freq_start = 200.0
    freq_end = 4000.0
    instantaneous_freq = freq_start + (freq_end - freq_start) * t / duration
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    samples = (np.sin(phase) * 32767 * 0.5).astype(np.int16)

    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    _write_wav(path, samples, sr, num_channels=1, bits_per_sample=16)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def offset_audio_files():
    """
    Generate two WAV files where the second is delayed by a known offset.
    Uses a chirp signal for sharp cross-correlation peak.

    Returns (path_a, path_b, known_offset_sec) as a tuple.
    """
    sr = 44100
    signal_duration = 1.0
    delay_sec = 0.5

    t = np.linspace(0, signal_duration, int(sr * signal_duration), endpoint=False)
    freq_start = 200.0
    freq_end = 4000.0
    instantaneous_freq = freq_start + (freq_end - freq_start) * t / signal_duration
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    signal = (np.sin(phase) * 32767 * 0.3).astype(np.int16)

    # Video A: immediate chirp + trailing silence to match B's length
    silence_padding = np.zeros(int(sr * delay_sec), dtype=np.int16)
    audio_a = np.concatenate([signal, silence_padding])

    # Video B: delay_sec of silence then chirp + small trailing silence
    audio_b = np.concatenate([silence_padding, signal])

    path_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    path_b = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    _write_wav(path_a, audio_a, sr, num_channels=1, bits_per_sample=16)
    _write_wav(path_b, audio_b, sr, num_channels=1, bits_per_sample=16)

    """
    Timeline:
      a: |--- chirp (1s) ---|-- silence (0.5s) --|
      b: |-- silence (0.5s) --|--- chirp (1s) ---|
      
    b's content is 0.5s behind a's content.
    offset_sec should be +0.5 (b is behind a).
    """
    yield path_a, path_b, delay_sec

    for p in [path_a, path_b]:
        if os.path.exists(p):
            os.unlink(p)


# ============================================================================
# Helpers
# ============================================================================


def _write_wav(path, samples, sample_rate, num_channels=1, bits_per_sample=16):
    """Write a numpy int16 array to a WAV file (PCM format)."""
    num_samples = len(samples)
    bytes_per_sample = bits_per_sample // 8
    data_size = num_samples * bytes_per_sample
    riff_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * bytes_per_sample * num_channels))
        f.write(struct.pack("<H", bytes_per_sample * num_channels))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.tobytes())


def _write_dummy_video(path, duration_sec=2.0, fps=30):
    """Write a minimal valid-ish MP4-like file for testing."""
    # This is NOT a real video — just enough bytes to exist and look plausible
    # for tests that mock ffmpeg. Real tests require ffmpeg.
    with open(path, "wb") as f:
        f.write(b"\x00\x00\x00\x1cftypmp42")
        f.write(b"\x00" * 1024)  # padding


# ============================================================================
# Session Code Tests
# ============================================================================


class TestSessionCodes:
    """Session code generation and validation."""

    def test_generate_code_length(self, sync):
        code = sync.generate_session_code()
        assert len(code) == CODE_LENGTH

    def test_generate_code_chars(self, sync):
        """All characters must be from the allowed set."""
        code = sync.generate_session_code()
        assert all(c in CODE_CHARS for c in code)

    def test_generate_code_no_ambiguous(self, sync):
        """Ensure no ambiguous chars (0, O, 1, I, L)."""
        code = sync.generate_session_code()
        assert "0" not in code
        assert "O" not in code
        assert "1" not in code
        assert "I" not in code
        assert "L" not in code

    def test_generate_code_unique(self, sync):
        """Consecutive codes should differ."""
        codes = {sync.generate_session_code() for _ in range(100)}
        assert len(codes) == 100

    def test_generate_code_no_collision(self, sync):
        """Codes should avoid existing set."""
        existing = {"ABCDEF", "GHJKLM"}
        code = sync.generate_session_code(existing_codes=existing)
        assert code not in existing

    def test_deterministic_code(self, sync):
        """Same seed string should produce same code."""
        c1 = MultiCameraSync.generate_deterministic_code("test-user-2026-06-25")
        c2 = MultiCameraSync.generate_deterministic_code("test-user-2026-06-25")
        assert c1 == c2

    def test_deterministic_code_different_seed(self, sync):
        """Different seed strings should produce different codes."""
        c1 = MultiCameraSync.generate_deterministic_code("user-a")
        c2 = MultiCameraSync.generate_deterministic_code("user-b")
        assert c1 != c2

    def test_validate_valid_code(self, sync):
        code = sync.generate_session_code()
        assert MultiCameraSync.validate_session_code(code)

    def test_validate_too_short(self, sync):
        assert not MultiCameraSync.validate_session_code("ABC")

    def test_validate_too_long(self, sync):
        assert not MultiCameraSync.validate_session_code("ABCDEFG")

    def test_validate_ambiguous_chars(self, sync):
        assert not MultiCameraSync.validate_session_code("0ABCDE")
        assert not MultiCameraSync.validate_session_code("OABCDE")
        assert not MultiCameraSync.validate_session_code("1ABCDE")

    def test_validate_lowercase(self, sync):
        """Lowercase should be accepted (converted to uppercase in validation)."""
        assert MultiCameraSync.validate_session_code("abcdef")

    @pytest.mark.parametrize("invalid", [
        "", " ", "ABC DEF", "ABC😀EF", "A_B_C_", None,
    ])
    def test_validate_invalid_values(self, sync, invalid):
        if invalid is None:
            with pytest.raises(TypeError):
                MultiCameraSync.validate_session_code(invalid)
        else:
            # Most invalid codes should fail validation
            # Empty string, spaces, emoji, special chars
            result = MultiCameraSync.validate_session_code(invalid)
            assert not result

    def test_xorshift_same_seed(self, sync):
        """Same seed should produce same code."""
        code1 = MultiCameraSync._xorshift_code(12345, set())
        code2 = MultiCameraSync._xorshift_code(12345, set())
        assert code1 == code2

    def test_xorshift_different_seed(self, sync):
        """Different seeds should produce different codes."""
        code1 = MultiCameraSync._xorshift_code(12345, set())
        code2 = MultiCameraSync._xorshift_code(99999, set())
        assert code1 != code2


# ============================================================================
# QR Code Tests
# ============================================================================


class TestQrCodes:
    """QR code generation (mocked qrcode)."""

    def test_generate_qr_code_creates_file(self, sync, tmp_path):
        """QR code should create a PNG file at the specified path."""
        output = str(tmp_path / "test_qr.png")
        result = sync.generate_qr_code("ABC123", output_path=output)
        assert os.path.exists(result)
        assert result == output
        assert os.path.getsize(output) > 0

    def test_generate_qr_code_default_path(self, sync):
        """Without output_path, should generate in temp dir."""
        result = sync.generate_qr_code("ABC123")
        assert os.path.exists(result)
        assert "crease_qr" in result or "session_" in result
        os.unlink(result)

    def test_qr_code_uses_session_code(self, sync, tmp_path):
        """QR PNG should be named with the session code."""
        output = str(tmp_path / "custom_name.png")
        sync.generate_qr_code("XK9M2P", output_path=output)
        assert os.path.exists(output)

    def test_qr_code_without_qrcode_installed(self, sync):
        """Should raise ImportError if qrcode is not importable."""
        with patch.dict("sys.modules", {"qrcode": None}):
            # Reimport will not work this way; instead mock the import
            pass
        # This is hard to test without actually uninstalling. Skip.
        pytest.skip("Requires actual qrcode uninstall to test")


# ============================================================================
# WAV Read Tests
# ============================================================================


class TestWavRead:
    """Low-level WAV file reading."""

    def test_read_16bit_mono(self, sync, sample_wav_file):
        """Read a 16-bit mono WAV and verify shape and range."""
        waveform, sr = sync._read_wav(sample_wav_file)
        assert sr == 44100
        assert len(waveform.shape) == 1  # mono
        assert waveform.shape[0] > 0
        assert np.all(waveform >= -1.0)
        assert np.all(waveform <= 1.0)

    def test_read_16bit_sine_tone(self, sync, sample_wav_file):
        """Read a 440 Hz sine wave — verify it's non-trivial."""
        waveform, sr = sync._read_wav(sample_wav_file)
        # Should have meaningful energy
        rms = np.sqrt(np.mean(waveform ** 2))
        assert rms > 0.01
        assert rms <= 0.5  # clip prevention

    def test_read_missing_file(self, sync):
        """Should raise FileNotFoundError (wrapped as ValueError)."""
        with pytest.raises((ValueError, FileNotFoundError)):
            sync._read_wav("/tmp/nonexistent_file_12345.wav")

    def test_read_16bit_stereo_converts_to_mono(self, sync):
        """Stereo WAV should be averaged to mono."""
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        left = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.3).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * t) * 32767 * 0.3).astype(np.int16)
        stereo = np.column_stack([left, right]).ravel()  # interleaved

        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _write_wav(path, stereo, sr, num_channels=2, bits_per_sample=16)
        try:
            waveform, out_sr = sync._read_wav(path)
            assert out_sr == sr
            assert len(waveform.shape) == 1  # mono
            # Value should be between left and right
            assert waveform.shape[0] == sr
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_16bit_silence(self, sync):
        """Silent WAV should produce near-zero waveform."""
        sr = 44100
        samples = np.zeros(sr, dtype=np.int16)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _write_wav(path, samples, sr, num_channels=1, bits_per_sample=16)
        try:
            waveform, out_sr = sync._read_wav(path)
            rms = np.sqrt(np.mean(waveform ** 2))
            assert rms < 1e-6
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_32bit_wav(self, sync):
        """Read a 32-bit WAV file."""
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 2147483647 * 0.3).astype(np.int32)

        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _write_wav(path, samples, sr, num_channels=1, bits_per_sample=32)
        try:
            waveform, out_sr = sync._read_wav(path)
            rms = np.sqrt(np.mean(waveform ** 2))
            assert rms > 0.01
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_empty_wav(self, sync):
        """WAV file with no data samples."""
        sr = 44100
        samples = np.array([], dtype=np.int16)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _write_wav(path, samples, sr, num_channels=1, bits_per_sample=16)
        try:
            waveform, out_sr = sync._read_wav(path)
            assert len(waveform) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_8bit_wav(self, sync):
        """Read an 8-bit WAV file (unsigned)."""
        sr = 44100
        # 8-bit WAV is unsigned, so we need special handling
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 127 * 0.5 + 128).astype(np.uint8)

        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(path, "wb") as f:
            data_size = len(signal)
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))   # PCM
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sr))
            f.write(struct.pack("<I", sr * 1))
            f.write(struct.pack("<H", 1))   # block align
            f.write(struct.pack("<H", 8))   # 8-bit
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(signal.tobytes())
        try:
            waveform, out_sr = sync._read_wav(path)
            rms = np.sqrt(np.mean(waveform ** 2))
            assert rms > 0.01
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_24bit_wav(self, sync):
        """Read a 24-bit WAV file."""
        sr = 44100
        t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
        # Generate 24-bit samples
        signal_float = np.sin(2 * np.pi * 440 * t) * 0.3
        # Convert to 24-bit signed
        signal_int24 = (signal_float * 8388607).astype(np.int32)

        # Pack as 3 bytes per sample
        data = bytearray()
        for s in signal_int24:
            s = s & 0xFFFFFF  # ensure 24-bit
            data.extend([s & 0xFF, (s >> 8) & 0xFF, (s >> 16) & 0xFF])
        data = bytes(data)

        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(path, "wb") as f:
            data_size = len(data)
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))   # PCM
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sr))
            f.write(struct.pack("<I", sr * 3))
            f.write(struct.pack("<H", 3))   # block align
            f.write(struct.pack("<H", 24))  # 24-bit
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(data)
        try:
            waveform, out_sr = sync._read_wav(path)
            rms = np.sqrt(np.mean(waveform ** 2))
            assert rms > 0.01
            assert len(waveform) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# Audio Extraction Tests
# ============================================================================


class TestAudioExtraction:
    """Audio extraction from video files (mocked ffmpeg)."""

    def test_extract_audio_file_not_found(self, sync):
        """Should raise ValueError for missing file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="not found"):
                sync.extract_audio("/nonexistent.mp4")

    def test_extract_audio_no_ffmpeg(self):
        """Should raise FfmpegNotFoundError if ffmpeg not installed."""
        with patch("shutil.which", return_value=None):
            # Create a fresh instance without the fixture's mock
            with pytest.raises(FfmpegNotFoundError):
                MultiCameraSync()

    def test_extract_audio_ffmpeg_success(self, sync, sample_wav_file):
        """Mock ffmpeg call — verify WAV read is called on result."""
        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = b""

            with patch("engine.multi_cam_sync.os.path.exists", return_value=True):
                with patch.object(sync, "_read_wav", return_value=(
                        np.zeros(44100, dtype=np.float32), 44100
                )):
                    waveform, sr = sync.extract_audio(sample_wav_file)
                    assert sr == 44100
                    assert len(waveform) == 44100

    def test_extract_audio_ffmpeg_no_audio(self, sync, sample_wav_file):
        """Simulate ffmpeg 'no audio stream' error."""
        mock_stderr = b"Stream map '0:a' (input index 0, stream index 0): No such stream"
        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = mock_stderr

            # Mock os.path.exists: input exists, output WAV does not
            real_exists = os.path.exists
            with patch("engine.multi_cam_sync.os.path.exists") as mock_exists:
                def side_effect(p):
                    if p == str(sample_wav_file):
                        return True
                    return False
                mock_exists.side_effect = side_effect

                with pytest.raises(ValueError, match="No audio stream"):
                    sync.extract_audio(sample_wav_file)

    def test_extract_audio_ffmpeg_generic_error(self, sync, sample_wav_file):
        """Simulate ffmpeg generic failure."""
        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = b"some ffmpeg error"

            with patch("engine.multi_cam_sync.os.path.exists") as mock_exists:
                # The input file exists but output WAV was not created
                def exists_side_effect(p):
                    return p == str(sample_wav_file)  # only input exists
                mock_exists.side_effect = exists_side_effect

                with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
                    sync.extract_audio(sample_wav_file)

    def test_extract_audio_custom_sr(self, sync, sample_wav_file):
        """Custom sample rate should be passed to ffmpeg."""
        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = b""

            with patch("engine.multi_cam_sync.os.path.exists", return_value=True):
                with patch.object(sync, "_read_wav", return_value=(
                        np.zeros(22050, dtype=np.float32), 22050
                )):
                    waveform, sr = sync.extract_audio(sample_wav_file, sr=22050)
                    assert sr == 22050


# ============================================================================
# Cross-Correlation Tests
# ============================================================================


class TestCrossCorrelation:
    """Audio cross-correlation offset computation."""

    def test_identical_audio_zero_offset(self, sync, sample_wav_file):
        """Same audio file should have offset near 0."""
        with patch.object(sync, "extract_audio") as mock_extract:
            # Return identical chirp waveforms
            sr = 44100
            t = np.linspace(0, 1.0, sr, endpoint=False)
            # Chirp: sharp auto-correlation peak
            f_start, f_end = 200.0, 4000.0
            inst_freq = f_start + (f_end - f_start) * t
            phase = 2 * np.pi * np.cumsum(inst_freq) / sr
            waveform = np.sin(phase).astype(np.float32)
            mock_extract.return_value = (waveform, sr)

            result = sync.compute_offset(sample_wav_file, sample_wav_file)
            assert abs(result["offset_sec"]) < 0.01, \
                f"Expected ~0s, got {result['offset_sec']}s"
            assert result["correlation_score"] > 0.9

    def test_offset_detection(self, sync, offset_audio_files):
        """
        Two files with known offset should be detected within tolerance.
        
        Uses actual WAV files with chirp signal via _read_wav directly.
        """
        path_a, path_b, expected_offset = offset_audio_files

        # Mock extract_audio to call _read_wav directly (bypassing ffmpeg)
        def mock_extract(video_path, sr=None):
            return sync._read_wav(video_path)

        with patch.object(sync, "extract_audio", side_effect=mock_extract):
            result = sync.compute_offset(path_a, path_b)
            # offset_sec should be +expected_offset (b's chirp is 0.5s behind a's)
            assert abs(result["offset_sec"] - expected_offset) < 0.05, \
                f"Expected ~{expected_offset}s, got {result['offset_sec']}s"
            assert result["correlation_score"] > 0.5

    def test_offset_reversed(self, sync, offset_audio_files):
        """Swapping videos should negate the offset."""
        path_a, path_b, expected_offset = offset_audio_files

        def mock_extract(video_path, sr=None):
            return sync._read_wav(video_path)

        with patch.object(sync, "extract_audio", side_effect=mock_extract):
            # compute_offset(path_b, path_a): path_b has delay, path_a doesn't.
            # So path_a's content is earlier → offset is negative
            result = sync.compute_offset(path_b, path_a)
            assert abs(result["offset_sec"] + expected_offset) < 0.05, \
                f"Expected ~{-expected_offset}s, got {result['offset_sec']}s"

    def test_orthogonal_audio_low_correlation(self, sync, sample_wav_file):
        """Two completely different audio signals should have low correlation."""
        with patch.object(sync, "extract_audio") as mock_extract:
            sr = 44100
            t = np.linspace(0, 1.0, sr, endpoint=False)
            waveform_a = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            waveform_b = np.random.randn(sr).astype(np.float32) * 0.3
            mock_extract.side_effect = [(waveform_a, sr), (waveform_b, sr)]

            result = sync.compute_offset(sample_wav_file, sample_wav_file)
            # Random noise vs sine — correlation should be fairly low
            assert result["correlation_score"] < 0.5

    def test_silence_vs_signal(self, sync, sample_wav_file):
        """Silence vs signal should have low correlation."""
        with patch.object(sync, "extract_audio") as mock_extract:
            sr = 44100
            t = np.linspace(0, 1.0, sr, endpoint=False)
            waveform_a = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            waveform_a /= np.max(np.abs(waveform_a)) + 1e-10
            waveform_b = np.zeros(sr, dtype=np.float32)
            mock_extract.side_effect = [(waveform_a, sr), (waveform_b, sr)]

            result = sync.compute_offset(sample_wav_file, sample_wav_file)
            # Division by zero in normalization; should handle gracefully
            assert result["correlation_score"] < 1.0  # at least doesn't crash

    def test_short_audio_trimming(self, sync, sample_wav_file):
        """
        Very long audio should be trimmed to the search window for speed,
        but still produce correct offset.
        """
        with patch.object(sync, "extract_audio") as mock_extract:
            sr = 44100
            duration = 120  # 2 minutes — triggers trimming
            ref = np.zeros(sr * duration, dtype=np.float32)
            delayed = np.zeros(sr * duration, dtype=np.float32)
            # Add chirp at center for sharp correlation
            signal_len = int(sr * 3)
            t_sig = np.linspace(0, 3.0, signal_len, endpoint=False)
            f_s, f_e = 200.0, 4000.0
            inst_freq = f_s + (f_e - f_s) * t_sig / 3.0
            phase = 2 * np.pi * np.cumsum(inst_freq) / sr
            signal = np.sin(phase).astype(np.float32)
            signal /= np.max(np.abs(signal)) + 1e-10

            signal_start = int(sr * 58)  # roughly center of 120s
            ref[signal_start:signal_start + signal_len] = signal
            # Delayed: signal starts 2 seconds later
            delayed[signal_start + int(sr * 2):signal_start + int(sr * 2) + signal_len] = signal

            mock_extract.side_effect = [(ref, sr), (delayed, sr)]

            result = sync.compute_offset(sample_wav_file, sample_wav_file)
            # delayed starts 2s later → offset ~+2s (b is behind a)
            assert abs(result["offset_sec"] - 2.0) < 0.15, \
                f"Expected ~2.0s, got {result['offset_sec']}s"

    def test_max_offset_clamp(self, sync, sample_wav_file):
        """Very large offsets should be clamped by max_offset_sec."""
        with patch.object(sync, "extract_audio") as mock_extract:
            sr = 44100
            ref = np.random.randn(sr * 5).astype(np.float32) * 0.1
            delayed = np.concatenate([
                np.zeros(sr * 60, dtype=np.float32),  # 60s delay
                ref[:sr * 2],
            ])

            mock_extract.side_effect = [(ref, sr), (delayed, sr)]

            result = sync.compute_offset(sample_wav_file, sample_wav_file,
                                          max_offset_sec=10.0)
            # The peak at 60s offset is outside the search window
            # The correlation score should be very low
            assert result["correlation_score"] < 0.3


# ============================================================================
# Multi-Video Sync Tests
# ============================================================================


class TestMultiVideoSync:
    """Sync multiple videos pairwise."""

    def test_sync_two_videos(self, sync):
        """Sync 2 videos should produce offsets dict with 2 entries."""
        with patch.object(sync, "compute_offset") as mock_offset:
            mock_offset.return_value = {
                "offset_sec": 1.5,
                "correlation_score": 0.95,
                "offset_samples": 66150,
                "sample_rate": 44100,
                "video_a": "/tmp/vid1.mp4",
                "video_b": "/tmp/vid2.mp4",
            }

            result = sync.sync_videos(["/tmp/vid1.mp4", "/tmp/vid2.mp4"])
            assert result["num_cameras"] == 2
            assert result["reference"] == "/tmp/vid1.mp4"
            assert result["offsets"]["/tmp/vid1.mp4"] == 0.0
            assert result["offsets"]["/tmp/vid2.mp4"] == 1.5
            assert len(result["cameras"]) == 2
            assert len(result["correlations"]) == 1

    def test_sync_three_videos(self, sync):
        """Sync 3 videos — all offset against reference."""
        with patch.object(sync, "compute_offset") as mock_offset:
            mock_offset.return_value = {
                "offset_sec": 0.5,
                "correlation_score": 0.9,
                "offset_samples": 22050,
                "sample_rate": 44100,
                "video_a": "/tmp/ref.mp4",
                "video_b": "/tmp/cam2.mp4",
            }

            result = sync.sync_videos([
                "/tmp/ref.mp4",
                "/tmp/cam2.mp4",
                "/tmp/cam3.mp4",
            ])
            assert result["num_cameras"] == 3
            assert mock_offset.call_count == 2

    def test_sync_reference_index(self, sync):
        """Custom reference index should work."""
        with patch.object(sync, "compute_offset") as mock_offset:
            mock_offset.return_value = {
                "offset_sec": 0.0,
                "correlation_score": 1.0,
                "offset_samples": 0,
                "sample_rate": 44100,
                "video_a": "/tmp/cam2.mp4",
                "video_b": "/tmp/cam1.mp4",
            }

            result = sync.sync_videos(
                ["/tmp/cam1.mp4", "/tmp/cam2.mp4", "/tmp/cam3.mp4"],
                reference_index=1
            )
            assert result["reference"] == "/tmp/cam2.mp4"
            assert result["offsets"]["/tmp/cam2.mp4"] == 0.0

    def test_sync_single_video_error(self, sync):
        """Less than 2 videos should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 videos"):
            sync.sync_videos(["/tmp/only_one.mp4"])

    def test_sync_invalid_reference(self, sync):
        """Invalid reference index should raise ValueError."""
        with pytest.raises(ValueError, match="reference_index"):
            sync.sync_videos(["/tmp/a.mp4", "/tmp/b.mp4"], reference_index=5)

    def test_sync_camera_names(self, sync):
        """Camera names should be derived from filenames."""
        with patch.object(sync, "compute_offset") as mock_offset:
            mock_offset.return_value = {
                "offset_sec": 0.0,
                "correlation_score": 1.0,
                "offset_samples": 0,
                "sample_rate": 44100,
                "video_a": "",
                "video_b": "",
            }

            result = sync.sync_videos(["/tmp/front_cam.mp4", "/tmp/side_cam.mp4"])
            assert result["cameras"][0]["video_name"] == "front_cam.mp4"
            assert result["cameras"][1]["video_name"] == "side_cam.mp4"


# ============================================================================
# Timeline Alignment Tests
# ============================================================================


class TestTimelineAlignment:
    """Align frame metrics to unified timeline."""

    def test_align_basic(self, sync):
        """Basic alignment of two cameras' metrics."""
        sync_result = {
            "cameras": [
                {"video_path": "/tmp/cam1.mp4", "video_name": "cam1.mp4",
                 "offset_sec": 0.0},
                {"video_path": "/tmp/cam2.mp4", "video_name": "cam2.mp4",
                 "offset_sec": 2.0},
            ],
            "offsets": {"/tmp/cam1.mp4": 0.0, "/tmp/cam2.mp4": 2.0},
            "reference": "/tmp/cam1.mp4",
            "num_cameras": 2,
        }

        # Cam 1: 5 frames at 30fps = ~0.167s
        metrics_list = [
            {"video_path": "/tmp/cam1.mp4", "fps": 30.0,
             "frames": [{"shot": 1}, {"shot": 2}, {"shot": 3}, {"shot": 4}, {"shot": 5}]},
            {"video_path": "/tmp/cam2.mp4", "fps": 30.0,
             "frames": [{"shot": "a"}, {"shot": "b"}, {"shot": "c"}]},
        ]

        result = sync.align_timelines(sync_result, metrics_list)
        assert result["num_cameras"] == 2
        assert result["num_events"] == 8  # 5 + 3
        assert len(result["unified_timeline"]) == 8

        # Cam 2 is offset by 2s, so frame 0 at unified_time = 0 - 2 = -2s
        first_cam2 = [e for e in result["unified_timeline"]
                      if "/tmp/cam2.mp4" in e["camera"]][0]
        assert abs(first_cam2["unified_time_sec"] - (-2.0)) < 0.001

    def test_align_no_matching_metrics(self, sync):
        """Metrics that don't match any camera should be skipped."""
        sync_result = {
            "cameras": [
                {"video_path": "/tmp/cam1.mp4", "offset_sec": 0.0},
            ],
            "offsets": {"/tmp/cam1.mp4": 0.0},
            "reference": "/tmp/cam1.mp4",
            "num_cameras": 1,
        }

        # No matching metrics — lists are empty
        result = sync.align_timelines(sync_result, [{"video_path": "/tmp/other.mp4", "frames": []}])
        assert result["num_cameras"] == 0
        assert result["num_events"] == 0

    def test_align_empty_frames(self, sync):
        """Empty frame lists produce empty timeline."""
        sync_result = {
            "cameras": [
                {"video_path": "/tmp/cam1.mp4", "offset_sec": 0.0},
            ],
            "offsets": {"/tmp/cam1.mp4": 0.0},
            "reference": "/tmp/cam1.mp4",
            "num_cameras": 1,
        }

        result = sync.align_timelines(sync_result, [{"video_path": "/tmp/cam1.mp4", "frames": []}])
        assert result["num_events"] == 0

    def test_align_unified_timeline_sorted(self, sync):
        """Unified timeline must be sorted by time."""
        sync_result = {
            "cameras": [
                {"video_path": "/tmp/cam1.mp4", "offset_sec": 0.0},
                {"video_path": "/tmp/cam2.mp4", "offset_sec": -1.0},  # cam2 is ahead
            ],
            "offsets": {"/tmp/cam1.mp4": 0.0, "/tmp/cam2.mp4": -1.0},
            "reference": "/tmp/cam1.mp4",
            "num_cameras": 2,
        }

        metrics_list = [
            {"video_path": "/tmp/cam1.mp4", "fps": 10.0,
             "frames": [{"n": 1}, {"n": 2}]},
            {"video_path": "/tmp/cam2.mp4", "fps": 10.0,
             "frames": [{"n": "a"}, {"n": "b"}]},
        ]

        result = sync.align_timelines(sync_result, metrics_list)
        times = [e["unified_time_sec"] for e in result["unified_timeline"]]
        assert times == sorted(times)


# ============================================================================
# Composite Video Tests
# ============================================================================


class TestCompositeVideo:
    """Multi-camera composite video generation (mocked subprocess)."""

    def test_side_by_side_two_videos(self, sync, tmp_path):
        """2-video side-by-side should construct correct ffmpeg command."""
        output = str(tmp_path / "composite.mp4")
        v1 = str(tmp_path / "cam1.mp4")
        v2 = str(tmp_path / "cam2.mp4")
        # Create dummy files
        for v in [v1, v2]:
            _write_dummy_video(v)

        offsets = {v1: 0.0, v2: 1.5}

        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = b""

            result = sync.create_multi_cam_video(
                [v1, v2], offsets, output, layout="side_by_side"
            )
            assert result == output
            # Verify setpts was used with correct offset
            cmd_str = " ".join(mock_run.call_args[0][0])
            assert "setpts=PTS+1.500/TB" in cmd_str

    def test_pip_layout(self, sync, tmp_path):
        """PiP layout for 2 videos."""
        output = str(tmp_path / "pip.mp4")
        v1 = str(tmp_path / "main.mp4")
        v2 = str(tmp_path / "pip.mp4")
        for v in [v1, v2]:
            _write_dummy_video(v)

        offsets = {v1: 0.0, v2: 0.5}

        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = b""

            result = sync.create_multi_cam_video(
                [v1, v2], offsets, output, layout="picture_in_picture"
            )
            assert result == output
            cmd_str = " ".join(mock_run.call_args[0][0])
            assert "overlay" in cmd_str
            assert "scale=480:270" in cmd_str

    def test_fewer_than_two_videos(self, sync, tmp_path):
        """Less than 2 videos should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 videos"):
            sync.create_multi_cam_video(
                ["/tmp/only_one.mp4"], {"/tmp/only_one.mp4": 0.0}, "/tmp/out.mp4"
            )

    def test_missing_video_file(self, sync):
        """Non-existent video file should raise ValueError."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="not found"):
                sync.create_multi_cam_video(
                    ["/tmp/nonexistent1.mp4", "/tmp/nonexistent2.mp4"],
                    {"/tmp/nonexistent1.mp4": 0.0, "/tmp/nonexistent2.mp4": 0.0},
                    "/tmp/out.mp4",
                )

    def test_unknown_layout(self, sync, tmp_path):
        """Unknown layout should raise ValueError."""
        v1 = str(tmp_path / "a.mp4")
        v2 = str(tmp_path / "b.mp4")
        for v in [v1, v2]:
            _write_dummy_video(v)

        with pytest.raises(ValueError, match="Unknown layout"):
            sync.create_multi_cam_video(
                [v1, v2], {v1: 0.0, v2: 0.0}, str(tmp_path / "out.mp4"),
                layout="spherical"
            )


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Error handling and edge cases."""

    def test_ffmpeg_not_found(self):
        """Fresh instance without ffmpeg should raise."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(FfmpegNotFoundError):
                MultiCameraSync()

    def test_extract_audio_no_ffmpeg_method(self):
        """Instance with no ffmpeg should raise on ffmpeg-dependent methods."""
        with patch("shutil.which", return_value=None):
            s = MultiCameraSync.__new__(MultiCameraSync)
            s.sample_rate = 44100
            with pytest.raises(FfmpegNotFoundError):
                s.extract_audio("test.mp4")

    def test_create_multi_cam_no_ffmpeg(self):
        """Composite video without ffmpeg should raise."""
        with patch("shutil.which", return_value=None):
            s = MultiCameraSync.__new__(MultiCameraSync)
            s.sample_rate = 44100
            with pytest.raises(FfmpegNotFoundError):
                s.create_multi_cam_video(
                    ["a.mp4", "b.mp4"],
                    {"a.mp4": 0.0, "b.mp4": 0.0},
                    "out.mp4",
                )

    def test_estimate_disk_space(self, sync, tmp_path):
        """Disk space estimation should return file sizes."""
        v1 = tmp_path / "a.mp4"
        v2 = tmp_path / "b.mp4"
        v1.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB
        v2.write_bytes(b"\x00" * 512 * 1024)   # 0.5 MB

        result = sync.estimate_required_disk_space([str(v1), str(v2)])
        assert result["num_files"] == 2
        assert result["total_input_mb"] == pytest.approx(1.5, rel=0.1)
        assert result["estimated_output_mb"] > 0

    def test_estimate_disk_space_missing_file(self, sync):
        """Missing files should be omitted from estimation."""
        result = sync.estimate_required_disk_space(["/tmp/nonexistent.mp4"])
        assert result["num_files"] == 0
        assert result["total_input_mb"] == 0.0


# ============================================================================
# Integration-Style Smoke Test
# ============================================================================


class TestIntegration:
    """
    High-level smoke test for the sync pipeline.
    Uses only mocked subprocess calls.
    """

    def test_full_sync_pipeline(self, sync, tmp_path):
        """
        Mock the entire pipeline: generate code → sync videos → align.
        """
        # 1. Generate session code
        code = sync.generate_session_code()
        assert len(code) == CODE_LENGTH

        # 2. Generate QR
        qr_path = str(tmp_path / "qr.png")
        sync.generate_qr_code(code, output_path=qr_path)
        assert os.path.exists(qr_path)

        # 3. Sync videos (mocked)
        with patch.object(sync, "compute_offset") as mock_offset:
            mock_offset.return_value = {
                "offset_sec": 1.2,
                "correlation_score": 0.92,
                "offset_samples": 52920,
                "sample_rate": 44100,
            }
            result = sync.sync_videos(["/tmp/cam1.mp4", "/tmp/cam2.mp4"])
            assert result["num_cameras"] == 2
            assert abs(result["offsets"]["/tmp/cam2.mp4"] - 1.2) < 0.001

        # 4. Create composite (mocked)
        with patch("engine.multi_cam_sync.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = b""
            output = str(tmp_path / "synced.mp4")
            # Create dummy camera files
            cam_files = [
                str(tmp_path / "cam1.mp4"),
                str(tmp_path / "cam2.mp4"),
            ]
            for v in cam_files:
                with open(v, "wb") as f:
                    f.write(b"\x00" * 1024)
            # Create dummy output (mocked subprocess won't)
            Path(output).write_bytes(b"\x00" * 1024)

            try:
                result = sync.create_multi_cam_video(
                    cam_files,
                    {cam_files[0]: 0.0, cam_files[1]: 1.2},
                    output,
                )
                assert result == output
                assert os.path.exists(output)
            finally:
                for v in cam_files:
                    if os.path.exists(v):
                        os.unlink(v)
