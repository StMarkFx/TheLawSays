#!/bin/bash
# Startup script for Railway deployment
# Ensures Python can find all packages and modules

set -e

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include project root and thelawsays_core
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/thelawsays_core:${PYTHONPATH}"

# Change to project root directory
cd "${SCRIPT_DIR}"

# Start the FastAPI application
exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"

