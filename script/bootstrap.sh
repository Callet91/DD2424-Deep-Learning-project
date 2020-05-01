#!/bin/sh
# Resolve all dependencies that the application requires to run.

# Stop on errors
set -e

cd "$(dirname "$0")/.."

echo "Installing development dependencies..."
pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt \
   && rm -rf /tmp/pip-tmp
