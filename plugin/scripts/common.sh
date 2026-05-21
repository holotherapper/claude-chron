#!/usr/bin/env bash
set -euo pipefail

CCHRON_CMD=""

_detect_cchron() {
  if command -v cchron &>/dev/null; then
    CCHRON_CMD="cchron"
    return
  fi
  for candidate in \
    "$HOME/.cargo/bin/cchron" \
    "/opt/homebrew/bin/cchron" \
    "/usr/local/bin/cchron"; do
    if [ -x "$candidate" ]; then
      CCHRON_CMD="$candidate"
      return
    fi
  done
}

_detect_cchron

_ensure_cchron() {
  # Never auto-install from a hook: that would be silent, unpinned network
  # access and code execution on every hook invocation. Tell the user instead.
  if [ -z "$CCHRON_CMD" ]; then
    echo '{"systemMessage": "[claude-chron] cchron not found. Install: cargo install --git https://github.com/holotherapper/claude-chron"}' >&1
    exit 0
  fi
}

# Read stdin with timeout (hooks pass JSON via stdin)
if command -v timeout &>/dev/null; then
  INPUT="$(timeout 2 cat 2>/dev/null || echo '{}')"
else
  INPUT="$(perl -e 'alarm 2; local $/; $_ = <STDIN>; print if defined' 2>/dev/null || echo '{}')"
fi
