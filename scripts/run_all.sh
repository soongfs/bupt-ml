#!/bin/bash
set -e
echo "=== Downloading ModelNet40 ==="
uv run pointcls download
echo "=== Training DGCNN ==="
uv run pointcls train --config configs/dgcnn.yaml
echo "=== Training PointMLP ==="
uv run pointcls train --config configs/pointmlp.yaml
echo "=== Done! Results in runs/ ==="
