#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Exits with an install hint if cchron is not found.
_ensure_cchron

# Pass hook input to cchron hook-stop via stdin
echo "$INPUT" | $CCHRON_CMD hook-stop 2>/dev/null || echo '{}'
