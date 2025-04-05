#!/bin/bash

# Exit on error
set -e

echo "Starting Jupyter Lab..."
uv run --with jupyter jupyter lab &

echo "Starting MLflow..."
uv run --with mlflow mlflow ui --host 0.0.0.0 --port 5000 &

wait
