#!/usr/bin/env bash
set -euo pipefail

# Create a standardized submission zip.
#
# Produces: code.zip

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Running public tests..."
uv run pytest

OUT="code.zip"
rm -f "$OUT"

echo "Creating $OUT..."
zip -j "$OUT" student.py >/dev/null

echo "Done: $OUT"
echo "Upload code.zip to Brightspace."
