#!/bin/sh
# Setups the repository.

# Stop on errors
set -e

echo "Starting setup of development environment..."
pip install --upgrade pip

echo "Install unzip..."
sudo apt install -y unzip
cd "$(dirname "$0")/.."
script/bootstrap.sh

pre-commit install

echo "Setup done!"
