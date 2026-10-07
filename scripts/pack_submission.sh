#!/usr/bin/env bash
# Build submission.zip with agent.yaml at the archive root (Kaggle requirement).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -f "$ROOT/submission.zip"
(cd "$ROOT/submission" && zip -qr "$ROOT/submission.zip" .)
unzip -l "$ROOT/submission.zip" | awk '{print $4}' | grep -qx agent.yaml \
  || { echo "error: agent.yaml not at zip root" >&2; exit 1; }
unzip -l "$ROOT/submission.zip"
