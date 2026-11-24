#!/bin/sh
# Use the caller's activated Python environment.
set -eu
exec python3 "$(dirname "$0")/verify_local.py" "$@"
