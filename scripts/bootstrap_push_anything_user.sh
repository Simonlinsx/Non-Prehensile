#!/usr/bin/env bash
set -euo pipefail

BAZELISK_VERSION="1.29.0"
BAZELISK_SHA256="5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992"
INSTALL_DIR="${PUSH_ANYTHING_USER_BIN:-/data1/linsixu/.local/bin}"
BAZELISK_PATH="$INSTALL_DIR/bazelisk"
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"

verify_bazelisk() {
  [[ -f "$BAZELISK_PATH" ]] || return 1
  local actual_sha
  actual_sha="$(sha256sum "$BAZELISK_PATH" | awk '{print $1}')"
  [[ "$actual_sha" == "$BAZELISK_SHA256" ]]
}

if verify_bazelisk; then
  echo "Bazelisk v${BAZELISK_VERSION} already installed: $BAZELISK_PATH"
  exit 0
fi

if [[ -e "$BAZELISK_PATH" ]]; then
  echo "ERROR: existing Bazelisk path has an unexpected checksum: $BAZELISK_PATH" >&2
  exit 2
fi

mkdir -p "$INSTALL_DIR"
temp_dir="$(mktemp -d /tmp/push-anything-bazelisk.XXXXXX)"
trap 'rm -rf "$temp_dir"' EXIT

curl -fL "$BAZELISK_URL" -o "$temp_dir/bazelisk"
echo "$BAZELISK_SHA256  $temp_dir/bazelisk" | sha256sum --check --strict
install -m 0755 "$temp_dir/bazelisk" "$BAZELISK_PATH"

echo "Installed Bazelisk v${BAZELISK_VERSION}: $BAZELISK_PATH"
echo "Bazel 8.4.0 will be downloaded into the user cache on the first build check."
