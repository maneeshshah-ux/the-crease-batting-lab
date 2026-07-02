"""
Cloud Storage — S3-compatible file persistence for Render deployments.

Supports:
  - Cloudflare R2 (set CLOUDFLARE_R2_* env vars)
  - Backblaze B2  (set B2_* env vars — B2 exposes an S3-compatible endpoint)
  - Local disk fallback when neither is configured

Activate by setting any of these in Render env vars:

    # Cloudflare R2
    CLOUDFLARE_R2_ENDPOINT  = https://<account_id>.r2.cloudflarestorage.com
    CLOUDFLARE_R2_ACCESS_KEY = ...
    CLOUDFLARE_R2_SECRET_KEY = ...
    CLOUDFLARE_R2_BUCKET     = the-crease-batting-lab

    # Backblaze B2 (S3-compatible mode)
    B2_ENDPOINT   = https://s3.<region>.backblazeb2.com
    B2_ACCESS_KEY = ...
    B2_SECRET_KEY = ...
    B2_BUCKET     = the-crease-batting-lab

    # Public CDN prefix for generated download URLs (optional)
    STORAGE_PUBLIC_URL_PREFIX = https://pub-<hash>.r2.dev

Files are always written to local disk first; cloud upload is async-friendly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Configuration detection ───────────────────────────────────────────────────

def _r2_config() -> Optional[dict]:
    endpoint = os.environ.get("CLOUDFLARE_R2_ENDPOINT", "")
    access   = os.environ.get("CLOUDFLARE_R2_ACCESS_KEY", "")
    secret   = os.environ.get("CLOUDFLARE_R2_SECRET_KEY", "")
    bucket   = os.environ.get("CLOUDFLARE_R2_BUCKET", "")
    if endpoint and access and secret and bucket:
        return {"endpoint": endpoint, "access": access,
                "secret": secret, "bucket": bucket}
    return None


def _b2_config() -> Optional[dict]:
    endpoint = os.environ.get("B2_ENDPOINT", "")
    access   = os.environ.get("B2_ACCESS_KEY", "")
    secret   = os.environ.get("B2_SECRET_KEY", "")
    bucket   = os.environ.get("B2_BUCKET", "")
    if endpoint and access and secret and bucket:
        return {"endpoint": endpoint, "access": access,
                "secret": secret, "bucket": bucket}
    return None


def is_cloud_storage_configured() -> bool:
    return bool(_r2_config() or _b2_config())


def _get_client_and_bucket():
    """Return (boto3 S3 client, bucket_name) or raise if not configured."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            "boto3 not installed. Add 'boto3' to requirements.txt to enable cloud storage."
        )

    cfg = _r2_config() or _b2_config()
    if not cfg:
        raise RuntimeError("No cloud storage configured. Set R2 or B2 env vars.")

    client = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access"],
        aws_secret_access_key=cfg["secret"],
        region_name="auto",
    )
    return client, cfg["bucket"]


# ── Public API ────────────────────────────────────────────────────────────────

def upload_file(local_path: str | Path, remote_key: str) -> Optional[str]:
    """
    Upload a local file to cloud storage.

    Args:
        local_path: Full local path to the file.
        remote_key: Key (path) inside the bucket, e.g. "sessions/abc123.json"

    Returns:
        Public URL string if STORAGE_PUBLIC_URL_PREFIX is set, else None.
        Returns None on failure (logs the error; caller continues with local disk).
    """
    if not os.path.isfile(local_path):
        log.warning("[CloudStorage] File not found, skipping upload: %s", local_path)
        return None

    if not is_cloud_storage_configured():
        return None  # Silently skip — local-disk mode

    try:
        client, bucket = _get_client_and_bucket()
        client.upload_file(str(local_path), bucket, remote_key)
        log.info("[CloudStorage] Uploaded %s → %s/%s", local_path, bucket, remote_key)

        prefix = os.environ.get("STORAGE_PUBLIC_URL_PREFIX", "").rstrip("/")
        if prefix:
            return f"{prefix}/{remote_key}"
        return None

    except Exception as exc:
        log.error("[CloudStorage] Upload failed for %s: %s", remote_key, exc)
        return None  # Non-fatal: local disk still has the file


def download_file(remote_key: str, local_path: str | Path) -> bool:
    """
    Download a file from cloud storage to local disk.

    Returns True on success, False on failure.
    """
    if not is_cloud_storage_configured():
        return False

    try:
        client, bucket = _get_client_and_bucket()
        os.makedirs(os.path.dirname(str(local_path)), exist_ok=True)
        client.download_file(bucket, remote_key, str(local_path))
        log.info("[CloudStorage] Downloaded %s → %s", remote_key, local_path)
        return True

    except Exception as exc:
        log.error("[CloudStorage] Download failed for %s: %s", remote_key, exc)
        return False


def session_remote_key(session_id: str) -> str:
    """Standard remote key for a session JSON."""
    return f"sessions/analysis_{session_id}.json"


def upload_session_json(session_id: str, local_path: str | Path) -> Optional[str]:
    """Upload a session analysis JSON to cloud storage."""
    return upload_file(local_path, session_remote_key(session_id))


def upload_report_pdf(session_id: str, local_path: str | Path) -> Optional[str]:
    """Upload a PDF report to cloud storage."""
    return upload_file(local_path, f"reports/report_{session_id}.pdf")


def upload_highlight_clip(
    session_id: str, clip_filename: str, local_path: str | Path
) -> Optional[str]:
    """Upload a highlight clip to cloud storage."""
    return upload_file(local_path, f"highlights/{session_id}/{clip_filename}")
