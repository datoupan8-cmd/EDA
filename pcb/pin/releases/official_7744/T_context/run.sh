#!/bin/sh
set -eu
exec python3 -B /submission/t_runtime/release_entry.py "$@"
