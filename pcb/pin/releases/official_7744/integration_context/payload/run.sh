#!/bin/sh
set -eu
exec python3 -B /submission/integration_runtime/entry.py "$@"
