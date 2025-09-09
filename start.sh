#!/usr/bin/env bash
set -e


# Paths used by app (can override with env vars in Render dashboard)
ASSET_DIR=${ASSET_DIR:-/tmp/assets}
FAISS_INDEX_PATH=${FAISS_INDEX_PATH:-$ASSET_DIR/faiss_index1.bin}
METADATA_PATH=${METADATA_PATH:-$ASSET_DIR/faiss_metadata1.json}
MEM_DB=${MEM_DB:-/tmp/memory.db}


# If ASSET_URL is set, download a zip/tar containing the assets and extract into ASSET_DIR
if [ ! -z "${ASSET_URL:-}" ]; then
echo "Downloading assets from ASSET_URL..."
mkdir -p "$ASSET_DIR"
curl -L --fail "$ASSET_URL" -o /tmp/assets_bundle || { echo "Failed to download assets"; exit 1; }
# Try unzip or tar
if file /tmp/assets_bundle | grep -q 'Zip archive'; then
unzip -o /tmp/assets_bundle -d "$ASSET_DIR"
else
tar -xzf /tmp/assets_bundle -C "$ASSET_DIR" || true
fi
fi


# Ensure the metadata path doesn't have accidental spaces or parentheses
if [ ! -f "$METADATA_PATH" ]; then
echo "Warning: METADATA_PATH '$METADATA_PATH' not found. Listing $ASSET_DIR:" >&2
ls -lah "$ASSET_DIR" || true
fi


export FAISS_INDEX_PATH=${FAISS_INDEX_PATH}
export METADATA_PATH=${METADATA_PATH}
export MEM_DB=${MEM_DB}


# Start gunicorn (Render sets $PORT automatically)
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
