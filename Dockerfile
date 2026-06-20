# =============================================================================
# Dockerfile — the CREASE Batting Lab
# =============================================================================
# Build:
#   docker build -t crease-batting-lab .
#
# Run locally:
#   docker run -p 5005:5005 crease-batting-lab
#
# Deploy to Render / Railway / Fly.io — just push this repo.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# ---------------------------------------------------------------------------
# Install system dependencies required by OpenCV, MediaPipe, and GLib
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Copy requirement files first (for Docker layer caching)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy the application code
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Create runtime directories (uploads, sessions, reports, frames)
# ---------------------------------------------------------------------------
RUN mkdir -p uploads sessions reports frames

# ---------------------------------------------------------------------------
# Run as a non-root user for security
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 crease && chown -R crease:crease /app
USER crease

# ---------------------------------------------------------------------------
# Expose the port (Render/Railway will set PORT env var)
# ---------------------------------------------------------------------------
EXPOSE 5005

# ---------------------------------------------------------------------------
# Start with Gunicorn (production-grade WSGI server)
# ---------------------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "--workers", "2", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
