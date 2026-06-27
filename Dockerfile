# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder — keeps image lean
COPY --from=builder /install /usr/local

# Copy application code only — never copy data/
COPY src/         ./src/
COPY artifacts/   ./artifacts/
COPY pyproject.toml .

# Install the src package in editable mode
RUN pip install -e . --no-deps

# Non-root user — security best practice
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check — Docker and compose use this to know when service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

 CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]