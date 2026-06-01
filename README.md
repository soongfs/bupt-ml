# BUPT ML Course — ModelNet40 Point Cloud Classification

DGCNN & PointMLP for 3D point cloud classification on ModelNet40.

## Setup

```bash
git clone <repo-url> bupt-ml
cd bupt-ml
uv sync
```

## Data

Download and extract the ModelNet40 dataset:

```bash
uv run pointcls download
```

This downloads from Princeton's server and extracts to `data/modelnet40/`.

## Training

Train DGCNN:

```bash
uv run pointcls train --config configs/dgcnn.yaml
```

Train PointMLP:

```bash
uv run pointcls train --config configs/pointmlp.yaml
```

Train both sequentially:

```bash
uv run pointcls train-all
```

Or use the shell script:

```bash
bash scripts/run_all.sh
```

Checkpoints and logs are saved to `runs/dgcnn/` and `runs/pointmlp/`.

## Testing

Run inference on test data with multi-view voting:

```bash
uv run pointcls test --checkpoint runs/dgcnn/best.pth --test-dir data/test --output result.csv
```

Test both models:

```bash
uv run pointcls test-both \
    --dgcnn-checkpoint runs/dgcnn/best.pth \
    --pointmlp-checkpoint runs/pointmlp/best.pth \
    --test-dir data/test \
    --output-dir results
```

## Server Deployment

To train on a remote server with GPU:

```bash
# On local machine
scp -r bupt-ml user@server:/path/to/

# On server
cd /path/to/bupt-ml
uv sync
uv run pointcls download
uv run pointcls train --config configs/dgcnn.yaml
```

## File Naming Convention for Submission

```
赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv
```

## Project Structure

```
bupt-ml/
├── pyproject.toml
├── README.md
├── configs/
│   ├── dgcnn.yaml
│   └── pointmlp.yaml
├── src/pointcls/
│   ├── cli.py
│   ├── train.py
│   ├── test.py
│   ├── data/
│   │   ├── download.py
│   │   ├── dataset.py
│   │   └── augment.py
│   └── models/
│       ├── dgcnn.py
│       └── pointmlp.py
├── docs/
│   ├── arch.md
│   ├── dgcnn.md
│   ├── pointmlp.md
│   └── design.md
└── scripts/
    └── run_all.sh
```
