#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Exits with an install hint if cchron is not found.
_ensure_cchron

echo "$INPUT" | $CCHRON_CMD hook-session-end 2>/dev/null || true
exit 0
