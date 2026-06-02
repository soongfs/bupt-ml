"""Test/inference module with voting strategy."""

import os
import csv
import time

import numpy as np
import torch

from pointcls.models import DGCNN, PointMLP


def run_test(
    checkpoint_path: str,
    test_dir: str,
    output_path: str,
    num_votes: int = 10,
    batch_size: int = 32,
):
    """Run inference on test data with multi-view voting.

    Args:
        checkpoint_path: Path to model checkpoint (.pth).
        test_dir: Directory containing test data.
            - If it contains .off/.txt files in subdirectories, use dataset loader.
            - If it contains .npy files, load them directly.
        output_path: Path to output CSV file.
        num_votes: Number of random rotations for voting.
        batch_size: Batch size for inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    best_acc = checkpoint.get("best_inst_acc", "N/A")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}, best inst acc: {best_acc}")

    # Instantiate model
    model_name = config.get("model", "")
    if not model_name:
        # Try to infer from checkpoint keys or path
        if "dgcnn" in checkpoint_path.lower():
            model_name = "dgcnn"
        elif "pointmlp" in checkpoint_path.lower():
            model_name = "pointmlp"
        else:
            raise ValueError("Cannot determine model type from checkpoint or path.")

    print(f"Model: {model_name}")

    if model_name == "dgcnn":
        model = DGCNN(
            k=config.get("k", 20),
            dropout=config.get("dropout", 0.5),
            num_classes=40,
            input_dim=6 if config.get("use_normals", False) else 3,
        )
    elif model_name == "pointmlp":
        model = PointMLP(
            num_classes=40,
            use_normals=config.get("use_normals", False),
            elite=config.get("elite", True),
            dropout=config.get("dropout", 0.5),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load test data
    use_normals = config.get("use_normals", False)
    num_points = config.get("num_points", 1024)
    test_samples, sample_ids = _load_test_data(
        test_dir,
        device,
        num_points=num_points,
        use_normals=use_normals,
    )
    print(f"Test samples: {len(test_samples)}")

    # Class names (alphabetically sorted, same as dataset)
    from pointcls.data.dataset import normalize_pointcloud
    # We know ModelNet40 has 40 classes, alphabetically sorted
    # Extract class names from dataset structure if possible
    class_names = _get_class_names(test_dir)

    # Inference with voting
    print(f"Running inference with {num_votes} votes...")
    predictions = []
    start_time = time.time()

    for i, points in enumerate(test_samples):
        # points: (N, 3) on device
        all_logits = []

        for v in range(num_votes):
            # Create rotated copy with different seed
            rotated = _random_rotate(points, seed=i * 1000 + v)

            # Normalize
            rotated = rotated.clone()
            rotated[:, :3] = normalize_pointcloud(rotated[:, :3])
            if not use_normals:
                rotated = rotated[:, :3]
            elif rotated.shape[1] < 6:
                pad = torch.zeros(
                    rotated.shape[0],
                    6 - rotated.shape[1],
                    dtype=rotated.dtype,
                    device=rotated.device,
                )
                rotated = torch.cat([rotated, pad], dim=1)

            # Prepare batch (1, 3, N)
            batch = rotated.unsqueeze(0).transpose(2, 1).contiguous()  # (1, C, N)

            with torch.no_grad():
                logits = model(batch)  # (1, 40)

            all_logits.append(logits)

        # Average logits across votes
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (1, 40)
        pred_class = avg_logits.argmax(dim=1).item()
        predictions.append(pred_class)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {i + 1}/{len(test_samples)} samples ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"Inference complete: {len(test_samples)} samples in {elapsed:.1f}s")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_class"])
        for sid, pred in zip(sample_ids, predictions):
            class_name = class_names[pred] if pred < len(class_names) else f"class_{pred}"
            writer.writerow([sid, class_name])

    print(f"Results saved to: {output_path}")


def _load_test_data(
    test_dir: str,
    device: torch.device,
    num_points: int = 1024,
    use_normals: bool = False,
):
    """Load test data from directory.

    Returns:
        test_samples: list of (N, C) tensors on device.
        sample_ids: list of sample identifiers.
    """
    test_samples = []
    sample_ids = []

    if _has_modelnet_test_split(test_dir):
        from pointcls.data.dataset import ModelNet40Dataset

        try:
            dataset = ModelNet40Dataset(
                root=test_dir,
                split="test",
                num_points=num_points,
                use_normals=use_normals,
                augment=False,
                preload=False,
            )
        except RuntimeError as exc:
            print(f"Could not load ModelNet40 test split: {exc}")
        else:
            print(f"Found {len(dataset)} ModelNet40 test files")
            for i in range(len(dataset)):
                points, _ = dataset[i]
                test_samples.append(points.to(device))
                sample_ids.append(os.path.splitext(os.path.basename(dataset.filepaths[i]))[0])
            return test_samples, sample_ids

    # Check for .npy files
    npy_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(".npy")
        and os.path.isfile(os.path.join(test_dir, f))
    ])
    if npy_files:
        print(f"Found {len(npy_files)} .npy files")
        for i, fname in enumerate(npy_files):
            data = np.load(os.path.join(test_dir, fname))
            points = _prepare_points(data, device, num_points, use_normals)
            test_samples.append(points)
            sample_ids.append(fname.replace(".npy", ""))
        return test_samples, sample_ids

    # Check for .off/.txt files
    point_files = []
    for root, dirs, files in os.walk(test_dir):
        for fname in sorted(files):
            if fname.lower().endswith((".off", ".txt")):
                point_files.append(os.path.join(root, fname))
    point_files = sorted(point_files)

    if point_files:
        print(f"Found {len(point_files)} .off/.txt files")
        for fpath in point_files:
            vertices = _read_point_file(fpath)
            points = _prepare_points(vertices, device, num_points, use_normals)
            test_samples.append(points)
            sample_ids.append(os.path.splitext(os.path.basename(fpath))[0])
        return test_samples, sample_ids

    raise FileNotFoundError(f"No .off, .txt, or .npy files found in {test_dir}")


def _has_modelnet_test_split(test_dir: str) -> bool:
    """Return True when root/class/test directories are present."""
    if not os.path.isdir(test_dir):
        return False
    for cls_name in os.listdir(test_dir):
        if cls_name.startswith((".", "_")) or cls_name == "__MACOSX":
            continue
        split_dir = os.path.join(test_dir, cls_name, "test")
        if os.path.isdir(split_dir):
            return True
    return False


def _read_point_file(filepath: str) -> np.ndarray:
    from pointcls.data.dataset import read_off, read_txt

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".off":
        return read_off(filepath)
    if ext == ".txt":
        return read_txt(filepath)
    raise ValueError(f"Unsupported point cloud file extension: {filepath}")


def _prepare_points(
    data: np.ndarray,
    device: torch.device,
    num_points: int,
    use_normals: bool,
) -> torch.Tensor:
    from pointcls.data.dataset import farthest_point_sample, normalize_pointcloud

    data = np.atleast_2d(data).astype(np.float32, copy=False)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least xyz columns, got shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Empty point cloud")

    points = torch.from_numpy(data).to(device)
    if points.shape[1] > 6:
        points = points[:, :6]
    if use_normals and points.shape[1] < 6:
        pad = torch.zeros(
            points.shape[0],
            6 - points.shape[1],
            dtype=points.dtype,
            device=points.device,
        )
        points = torch.cat([points, pad], dim=1)
    elif not use_normals:
        points = points[:, :3]

    if points.shape[0] > num_points:
        batch = points[:, :3].unsqueeze(0)
        fps_idx = farthest_point_sample(batch, num_points).squeeze(0)
        points = points[fps_idx]
    elif points.shape[0] < num_points:
        repeat = num_points // points.shape[0] + 1
        points = points.repeat(repeat, 1)[:num_points]

    points = points.clone()
    points[:, :3] = normalize_pointcloud(points[:, :3])
    return points


def _get_class_names(test_dir: str):
    """Try to infer class names from directory structure, else return default list."""
    # Default ModelNet40 class names (alphabetical)
    default = [
        "airplane", "bathtub", "bed", "bench", "bookshelf",
        "bottle", "bowl", "car", "chair", "cone",
        "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
        "laptop", "mantel", "monitor", "night_stand", "person",
        "piano", "plant", "radio", "range_hood", "sink",
        "sofa", "stairs", "stool", "table", "tent",
        "toilet", "tv_stand", "vase", "wardrobe", "xbox",
    ]

    # Try to read from data/modelnet40
    try:
        import os
        data_dir = "data/modelnet40"
        if os.path.isdir(data_dir):
            classes = sorted([
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ])
            if len(classes) == 40:
                return classes
    except Exception:
        pass

    return default


def _random_rotate(points: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply random SO(3) rotation with fixed seed.

    Args:
        points: (N, 3) tensor.
        seed: Random seed for reproducibility.

    Returns:
        Rotated points (N, 3).
    """
    from scipy.spatial.transform import Rotation

    # Use a local random state with the given seed
    rng = np.random.RandomState(seed)
    R = torch.from_numpy(
        Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    ).to(points.device)

    rotated = points.clone()
    rotated[:, :3] = points[:, :3] @ R.T
    if points.shape[1] >= 6:
        rotated[:, 3:6] = points[:, 3:6] @ R.T
    return rotated
