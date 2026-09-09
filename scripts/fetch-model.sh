#!/usr/bin/env bash
#
# Download the trained SavedModel into ./cnn-models/.
#
# Model weights are NOT stored in git (they are ~88 MB). They ship as a
# GitHub Release asset instead. This script fetches and unpacks them.
#
# Usage:
#   scripts/fetch-model.sh [RELEASE_TAG]
#
# Requires: gh (authenticated) or curl for a public repo.

set -euo pipefail

REPO="${MODEL_REPO:-MoleCare/molecare-ml}"
TAG="${1:-${MODEL_RELEASE_TAG:-weights-v1}}"
ASSET="${MODEL_ASSET:-xception-savedmodel-v1.tar.gz}"
DEST="${MODEL_DEST:-cnn-models}"

if [ -d "$DEST/xception/1/variables" ]; then
  echo "Model already present at $DEST/xception/1 — nothing to do."
  exit 0
fi

echo "Fetching $ASSET from $REPO@$TAG ..."
mkdir -p "$DEST"

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --output - \
    | tar -xzf - -C "$DEST"
else
  curl -fsSL "https://github.com/$REPO/releases/download/$TAG/$ASSET" \
    | tar -xzf - -C "$DEST"
fi

if [ ! -d "$DEST/xception/1/variables" ]; then
  echo "ERROR: expected $DEST/xception/1/variables after unpacking $ASSET" >&2
  exit 1
fi

echo "Model ready at $DEST/xception/1"
