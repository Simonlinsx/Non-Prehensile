#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C3_ROOT="${1:-/data1/linsixu/c3-push-anything}"
EXPECTED_COMMIT="5c08cb2e14b1ab10e024cb46e8504970cffcd5ea"
PATCH="$REPO_ROOT/third_party/push_anything/patches/0002-c3-no-gurobi-optional-m.patch"

if [[ ! -d "$C3_ROOT/.git" ]]; then
  echo "C3 checkout not found: $C3_ROOT" >&2
  exit 2
fi

actual_commit="$(git -C "$C3_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Expected C3 $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 3
fi

if git -C "$C3_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "Patch already applied: $PATCH"
  exit 0
fi

git -C "$C3_ROOT" apply --check "$PATCH"
git -C "$C3_ROOT" apply "$PATCH"
echo "Applied: $PATCH"
