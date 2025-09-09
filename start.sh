#!/usr/bin/env bash
set -euo pipefail

# Optional pre-start tasks could go here (migrations etc.)
# e.g. echo "Running migrations..."

# If FAISS index path or metadata are mounted into /app/data, nothing else to do.
# Execute whatever CMD is provided in the Dockerfile/CMD
exec "$@"
