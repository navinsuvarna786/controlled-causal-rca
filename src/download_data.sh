#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_URL="https://example.com/path/to/aiops_challenge_2020_stage1.zip" ./download_data.sh
# Optional:
#   DATA_SHA256="<expected_sha256>" DATA_URL="..." ./download_data.sh

ZIP_NAME="aiops_challenge_2020_stage1.zip"
OUT_DIR="aiops_challenge_2020_stage1"

if [[ -z "${DATA_URL:-}" ]]; then
  echo "ERROR: DATA_URL is not set."
  echo "Example: DATA_URL=\"https://example.com/${ZIP_NAME}\" ./download_data.sh"
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  echo "Downloading with curl..."
  curl -fL --retry 3 --retry-delay 2 -o "$ZIP_NAME" "$DATA_URL"
elif command -v wget >/dev/null 2>&1; then
  echo "Downloading with wget..."
  wget -O "$ZIP_NAME" "$DATA_URL"
else
  echo "ERROR: neither curl nor wget is installed."
  exit 1
fi

if [[ -n "${DATA_SHA256:-}" ]]; then
  echo "Verifying SHA256..."
  echo "${DATA_SHA256}  ${ZIP_NAME}" | sha256sum -c -
fi

mkdir -p "$OUT_DIR"
echo "Extracting ${ZIP_NAME}..."
unzip -o "$ZIP_NAME" -d "$OUT_DIR"

echo "Done."
echo "Archive: ${ZIP_NAME}"
echo "Extracted to: ${OUT_DIR}/"
