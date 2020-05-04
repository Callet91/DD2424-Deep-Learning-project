#!/bin/sh
# Resolve all dependencies that the application requires to run.

# Stop on errors
set -e

cd "$(dirname "$0")/.."

echo "Installing development dependencies..."
pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt \
   && rm -rf /tmp/pip-tmp

echo "Dependencies installed succesfully!"

echo "Setting local path for Python3..."

export PYTHONPATH=$PWD

echo "Path for python is now:" $PYTHONPATH
