#!/bin/bash
# Entrypoint for the Databricks App runtime.
#
# Imports inside backend/ are bare (e.g. `from routes.agent import ...`),
# so we must cd into backend/ before launching uvicorn. Apps inject the
# bind port via DATABRICKS_APP_PORT (typically 8000); fall back to 7860
# for local dev when the var is unset.

set -e
cd "$(dirname "$0")"
PORT="${DATABRICKS_APP_PORT:-7860}"
exec python -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
