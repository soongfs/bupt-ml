"""Training module for point cloud classification."""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from pointcls.data.dataset import ModelNet40Dataset
from pointcls.models.factory import build_model, infer_model_name


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute instance accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def compute_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 40) -> float:
    """Compute per-class mean accuracy."""
    preds = logits.argmax(dim=1)
    acc_per_class = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            acc_per_class.append((preds[mask] == c).float().mean().item())
    if len(acc_per_class) == 0:
        return 0.0
    return float(np.mean(acc_per_class))


def evaluate(model, dataloader, device, num_classes=40, desc: str = "Evaluating"):
    """Evaluate model on a dataloader.

    Returns:
        inst_acc: Instance accuracy (correct / total).
        class_acc: Per-class mean accuracy.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        with tqdm(dataloader, desc=desc, leave=False) as progress:
            for points, labels in progress:
                points, labels = points.to(device), labels.to(device)
                # Dataset returns (B, N, C) — transpose to (B, C, N)
                points = points.transpose(2, 1).contiguous()
                logits = model(points)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    inst_acc = compute_accuracy(all_logits, all_labels)
    class_acc = compute_class_accuracy(all_logits, all_labels, num_classes)

    return inst_acc, class_acc


def train_model(config_path: str, overrides: dict | None = None):
    """Train a model according to a config file.

    Args:
        config_path: Path to YAML config file.
        overrides: Optional dict of config values to override (e.g., {"batch_size": 2, "epochs": 3}).

    Returns:
        Tuple of (best_inst_acc, best_class_acc).
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            config[k] = v
        print(f"Config overrides applied: {overrides}")

    model_name = config["model"]
    output_dir = os.path.join("runs", model_name)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    resume_checkpoint = None
    if os.path.exists(checkpoint_path):
        resume_checkpoint = _load_training_checkpoint(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Reserve GPU memory BEFORE the slow CPU preload, so other users
    # on shared machines cannot claim this GPU while we prepare data.
    _gpu_reservation = None
    if device.type == "cuda":
        reserve_mb = config.get("gpu_reserve_mb", 0)
        if reserve_mb > 0:
            _gpu_reservation = _reserve_gpu_memory(reserve_mb)

    # Data
    data_dir = "data/modelnet40"
    train_dataset = ModelNet40Dataset(
        root=data_dir,
        split="train",
        num_points=config.get("num_points", 1024),
        use_normals=config.get("use_normals", False),
        augment=True,
        rotation_mode=config.get("rotation_mode", config.get("rotation_aug", "z")),
    )
    test_dataset = ModelNet40Dataset(
        root=data_dir,
        split="test",
        num_points=config.get("num_points", 1024),
        use_normals=config.get("use_normals", False),
        augment=False,
    )

    if _gpu_reservation is not None:
        del _gpu_reservation
        torch.cuda.empty_cache()
        print("Released GPU memory reservation before training.")

    print(
        f"Dataset layout: {train_dataset.layout}; "
        f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
    )

    num_workers = config.get("num_workers", 4)
    pin_memory = device.type == "cuda"
    prefetch = config.get("prefetch_factor", 2) if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = build_model(config)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} CUDA devices.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")

    # Loss
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.2)
    )

    # Optimizer
    optimizer_name = config.get("optimizer", "Adam")
    lr = config.get("lr", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("scheduler_T_max", 250),
    )

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training tracking
    epochs = config.get("epochs", 250)
    scheduler_t_max = config.get("scheduler_T_max", 250)
    start_epoch = 1
    best_inst_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    elapsed_time_prior = 0.0

    history = _initial_history()

    if resume_checkpoint is not None:
        _unwrap_model(model).load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if getattr(scheduler, "T_max", scheduler_t_max) != scheduler_t_max:
            print(
                "Scheduler T_max changed from "
                f"{scheduler.T_max} to {scheduler_t_max}; continuing with updated schedule."
            )
            scheduler.T_max = scheduler_t_max

        checkpoint_epoch = int(resume_checkpoint["epoch"])
        start_epoch = checkpoint_epoch + 1
        best_inst_acc = float(resume_checkpoint.get("best_inst_acc", best_inst_acc))
        best_class_acc = float(resume_checkpoint.get("best_class_acc", best_class_acc))
        history = _normalize_history(resume_checkpoint.get("history", {}))
        best_epoch = int(
            resume_checkpoint.get("best_epoch")
            or _infer_best_epoch(history, fallback=checkpoint_epoch)
        )
        elapsed_time_prior = float(resume_checkpoint.get("elapsed_time_total", 0.0))
        print(f"Resuming from epoch {start_epoch} (best_inst_acc={best_inst_acc:.4f})")

    # Log file
    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "a" if resume_checkpoint is not None else "w")
    if resume_checkpoint is not None:
        log_file.write(
            f"\nResuming from epoch {start_epoch} "
            f"(best_inst_acc={best_inst_acc:.4f})\n"
        )
        log_file.flush()

    start_time = time.time()
    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, epochs + 1):
            # Training
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            with tqdm(
                train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False
            ) as progress:
                for points, labels in progress:
                    points, labels = points.to(device), labels.to(device)
                    points = points.transpose(2, 1).contiguous()  # (B, N, C) -> (B, C, N)

                    optimizer.zero_grad()
                    logits = model(points)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += batch_size
                    progress.set_postfix(
                        loss=total_loss / max(total_samples, 1),
                        acc=total_correct / max(total_samples, 1),
                    )

            scheduler.step()

            if total_samples == 0:
                raise RuntimeError(
                    "No training samples were processed. Reduce batch_size or check the dataset."
                )

            avg_loss = total_loss / total_samples
            train_inst = total_correct / total_samples

            # Evaluate
            train_eval_inst, train_eval_class = evaluate(
                model, train_loader, device, desc="Train eval"
            )
            test_inst, test_class = evaluate(
                model, test_loader, device, desc="Test eval"
            )

            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            log_line = (
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_eval_inst:.4f} | "
                f"Test Inst Acc: {test_inst:.4f} | "
                f"Test Class Acc: {test_class:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            # Track history
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_eval_inst)
            history["train_inst_acc"].append(train_eval_inst)
            history["train_class_acc"].append(train_eval_class)
            history["test_inst_acc"].append(test_inst)
            history["test_class_acc"].append(test_class)

            # Save best
            if test_inst > best_inst_acc:
                best_inst_acc = test_inst
                best_class_acc = test_class
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": _unwrap_model(model).state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "best_inst_acc": best_inst_acc,
                        "best_class_acc": best_class_acc,
                    },
                    os.path.join(output_dir, "best.pth"),
                )
                print(f"  -> New best model saved! (inst_acc: {best_inst_acc:.4f})")

            last_completed_epoch = epoch
            _save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_inst_acc=best_inst_acc,
                best_class_acc=best_class_acc,
                best_epoch=best_epoch,
                history=history,
                config=config,
                seed=seed,
                elapsed_time_total=elapsed_time_prior + time.time() - start_time,
            )

        elapsed = elapsed_time_prior + time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")
        print(
            f"Best: Epoch {best_epoch}, Inst Acc: {best_inst_acc:.4f}, "
            f"Class Acc: {best_class_acc:.4f}"
        )

        log_file.write(
            f"\nBest: Epoch {best_epoch}, Inst Acc: {best_inst_acc:.4f}, "
            f"Class Acc: {best_class_acc:.4f}\n"
        )
        log_file.flush()

        # Save last checkpoint
        torch.save(
            {
                "model_state_dict": _unwrap_model(model).state_dict(),
                "config": config,
                "epoch": last_completed_epoch,
                "best_inst_acc": best_inst_acc,
                "best_class_acc": best_class_acc,
            },
            os.path.join(output_dir, "last.pth"),
        )

        # Plot training curves
        _plot_curves(history, output_dir)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return best_inst_acc, best_class_acc
    except KeyboardInterrupt:
        print("\nTraining interrupted. Resume by rerunning the same command.")
        log_file.write("\nTraining interrupted. Resume by rerunning the same command.\n")
        log_file.flush()
        raise
    finally:
        log_file.close()
        _shutdown_dataloader_workers(train_loader)
        _shutdown_dataloader_workers(test_loader)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _plot_curves(history: dict, output_dir: str):
    """Plot and save training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Train loss
    axes[0].plot(epochs, history["train_loss"], "b-", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    # Test instance accuracy
    axes[1].plot(epochs, history["test_inst_acc"], "g-", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Test Instance Accuracy")
    axes[1].grid(True)

    # Test class accuracy
    axes[2].plot(epochs, history["test_class_acc"], "r-", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Test Class Accuracy")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=100)
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")


def _initial_history() -> dict:
    return {
        "train_loss": [],
        "train_acc": [],
        "train_inst_acc": [],
        "train_class_acc": [],
        "test_inst_acc": [],
        "test_class_acc": [],
    }


def _normalize_history(raw: dict) -> dict:
    h = _initial_history()
    for key in h:
        if key in raw and isinstance(raw[key], list):
            h[key] = [float(v) for v in raw[key]]
    return h


def _infer_best_epoch(history: dict, fallback: int) -> int:
    if not history.get("test_inst_acc"):
        return fallback
    return int(history["test_inst_acc"].index(max(history["test_inst_acc"]))) + 1


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _load_training_checkpoint(path: str) -> dict:
    print(f"Found training checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_training_checkpoint(
    checkpoint_path, model, optimizer, scheduler, epoch,
    best_inst_acc, best_class_acc, best_epoch, history,
    config, seed, elapsed_time_total,
):
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_inst_acc": best_inst_acc,
            "best_class_acc": best_class_acc,
            "best_epoch": best_epoch,
            "history": history,
            "config": config,
            "seed": seed,
            "elapsed_time_total": elapsed_time_total,
        },
        checkpoint_path,
    )


def _shutdown_dataloader_workers(loader):
    if hasattr(loader, "_iterator") and loader._iterator is not None:
        loader._iterator._shutdown_workers()


def _print_cuda_info():
    """Print free/completed memory per GPU so the user can pick a free one."""
    if not torch.cuda.is_available():
        return
    print("GPU memory status:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        pct = used / total * 100
        print(
            f"  GPU {i} ({props.name}): "
            f"{used / 1024**3:.1f} / {total / 1024**3:.1f} GiB used ({pct:.0f}%)"
        )


def _reserve_gpu_memory(mb: int):
    """Allocate a tensor to reserve GPU memory, preventing other processes
    from claiming the GPU during long CPU preloads."""
    elements = (mb * 1024 * 1024) // 4  # float32 = 4 bytes
    t = torch.zeros(elements, dtype=torch.float32, device="cuda")
    print(f"Reserved {mb} MiB GPU memory during data preload.")
    return t
