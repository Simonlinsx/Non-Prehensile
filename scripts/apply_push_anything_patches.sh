#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${1:-/data1/linsixu/dairlib-push-anything}"
EXPECTED_COMMIT="9d988c835d6e99330397701487fce5ce4ceafa3c"
PATCH="$REPO_ROOT/third_party/push_anything/patches/0001-safe-only-sampling-mesh.patch"

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
  echo "Push Anything checkout not found: $UPSTREAM_ROOT" >&2
  exit 2
fi

ACTUAL_COMMIT="$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Expected upstream $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
  exit 3
fi

if git -C "$UPSTREAM_ROOT" apply --unidiff-zero --reverse --check "$PATCH" 2>/dev/null; then
  echo "Patch already applied: $PATCH"
  exit 0
fi

git -C "$UPSTREAM_ROOT" apply --unidiff-zero --check "$PATCH"
git -C "$UPSTREAM_ROOT" apply --unidiff-zero "$PATCH"
echo "Applied: $PATCH"
