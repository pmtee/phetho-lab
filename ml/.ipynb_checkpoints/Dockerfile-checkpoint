#── Base image ──────────────────────────────────
# python:3.11-slim = Python 3.11 on minimal Debian
# slim = smaller image, no unnecessary packages
FROM python:3.11-slim

#  ── Set working directory ────────────────────────
# All subsequent commands run from /app
WORKDIR /app

# ── Copy requirements first ──────────────────────
# We copy requirements.txt before the rest of code
# Docker caches each layer — if requirements dont
# change, pip install is skipped on rebuild
# This makes rebuilds much faster
COPY requirements.txt .

# ── Install dependencies ─────────────────────────
# --no-cache-dir = dont cache pip files (smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ───────────────────────
# Copy main.py and the saved model
COPY main.py .
COPY models/ ./models/

# ── Document the port ────────────────────────────
# EXPOSE documents which port the app listens on
# Does not actually publish the port — that happens
# when you run the container with -p flag
EXPOSE 8000

# ── Start command ────────────────────────────────
# CMD runs when container starts
# We use uvicorn directly — not python main.py
# host 0.0.0.0 = accept connections from outside
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
