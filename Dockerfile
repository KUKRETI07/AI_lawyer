# ---- build stage ----
FROM python:3.11-slim AS build
WORKDIR /app

# Install build deps required for some Python wheels
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential curl git wget ca-certificates libgomp1 libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for layer caching
COPY requirements.txt ./

# Create and use virtualenv so we can copy a clean env into final image
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip then install dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ---- final stage ----
FROM python:3.11-slim
WORKDIR /app

# runtime deps needed for some packages (kept minimal)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl ca-certificates libgomp1 libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtualenv from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code
COPY . /app

# Create app data dir (for mounting index files)
RUN mkdir -p /app/data && chown -R appuser:appuser /app /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
EXPOSE $PORT

USER appuser

# healthcheck (optional - requires curl to be present in runtime; we used curl above)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:$PORT/health || exit 1

# Use start script (provided below) which executes gunicorn
ENTRYPOINT ["/app/start.sh"]
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "3", "--access-logfile", "-", "--error-logfile", "-"]
