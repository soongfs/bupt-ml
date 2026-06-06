"""Logits ensembling for point cloud classifiers."""

from __future__ import annotations

import csv
import time

import torch

from pointcls.models.factory import build_model, infer_model_name
from pointcls.test import _get_class_names, _load_test_data, predict_logits_batched


def weighted_average_logits(
    models: list[torch.nn.Module],
    batch: torch.Tensor,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Forward all models and return weighted-average logits."""
    if not models:
        raise ValueError("At least one model is required")
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    if len(weights) != len(models):
        raise ValueError("weights length must match models length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    norm_weights = [w / total for w in weights]

    out = None
    for model, weight in zip(models, norm_weights):
        logits = model(batch)
        out = logits * weight if out is None else out + logits * weight
    return out


def load_checkpoint_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_name = infer_model_name(config, checkpoint_path)
    model = build_model(config, checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, config, model_name


def run_ensemble(
    checkpoint_paths: list[str],
    test_dir: str,
    output_path: str,
    weights: list[float] | None = None,
    num_votes: int = 1,
    rotation_mode: str = "none",
    batch_size: int = 32,
):
    """Run weighted logits ensemble on a labeled or unlabeled test directory."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    loaded = [load_checkpoint_model(path, device) for path in checkpoint_paths]
    models = [item[0] for item in loaded]
    configs = [item[1] for item in loaded]
    names = [item[2] for item in loaded]
    print("Models: " + ", ".join(names))

    use_normals = any(cfg.get("use_normals", False) for cfg in configs)
    num_points = max(int(cfg.get("num_points", 1024)) for cfg in configs)
    test_samples, sample_ids, labels = _load_test_data(
        test_dir,
        device,
        num_points=num_points,
        use_normals=use_normals,
    )
    print(f"Test samples: {len(test_samples)}")
    class_names = _get_class_names(test_dir)

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    total = float(sum(weights))
    weights = [w / total for w in weights]
    print(f"Running ensemble votes={num_votes}, rotation_mode={rotation_mode}, weights={weights}")

    start_time = time.time()
    ensemble_logits = None
    for model, cfg, weight in zip(models, configs, weights):
        logits = predict_logits_batched(
            model,
            test_samples,
            use_normals=cfg.get("use_normals", False),
            num_votes=num_votes,
            rotation_mode=rotation_mode,
            batch_size=batch_size,
        )
        ensemble_logits = logits * weight if ensemble_logits is None else ensemble_logits + logits * weight
    predictions = ensemble_logits.argmax(dim=1).tolist()
    elapsed = time.time() - start_time
    print(f"Ensemble inference complete: {len(test_samples)} samples in {elapsed:.1f}s")

    if labels is not None:
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        inst_acc = correct / len(labels)
        from collections import defaultdict
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for pred, label in zip(predictions, labels):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        per_class = [class_correct[c] / class_total[c] for c in sorted(class_total)]
        class_acc = sum(per_class) / len(per_class)
        print(f"Inst Acc: {correct}/{len(labels)} = {inst_acc:.4f}")
        print(f"Class Acc: {class_acc:.4f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_class"])
        for sid, pred in zip(sample_ids, predictions):
            class_name = class_names[pred] if pred < len(class_names) else f"class_{pred}"
            writer.writerow([sid, class_name])
    print(f"Results saved to: {output_path}")
