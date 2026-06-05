"""Unified CLI for point cloud classification."""

import click


@click.group()
def cli():
    """Point Cloud Classification on ModelNet40 — DGCNN & PointMLP."""
    pass


@cli.command()
@click.option(
    "--data-dir",
    default="data/modelnet40",
    help="Directory to download/extract ModelNet40.",
    show_default=True,
)
def download(data_dir: str):
    """Download and extract ModelNet40 dataset."""
    from pointcls.data.download import download_modelnet40
    download_modelnet40(data_dir)


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    help="Path to YAML config file (e.g., configs/dgcnn.yaml).",
)
@click.option("--batch-size", type=int, default=None, help="Override batch_size in config.")
@click.option("--epochs", type=int, default=None, help="Override epochs in config.")
def train(config: str, batch_size: int | None, epochs: int | None):
    """Train a model according to a config file."""
    from pointcls.train import train_model
    overrides = {}
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if epochs is not None:
        overrides["epochs"] = epochs
        overrides["scheduler_T_max"] = epochs
    train_model(config, overrides)


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    help="Path to model checkpoint (.pth).",
)
@click.option(
    "--test-dir",
    "-t",
    required=True,
    help="Directory containing test data (.off or .npy files).",
)
@click.option(
    "--output",
    "-o",
    default="result.csv",
    help="Output CSV path.",
    show_default=True,
)
@click.option(
    "--num-votes",
    default=1,
    help="Number of votes to average.",
    show_default=True,
)
@click.option(
    "--rotation-mode",
    type=click.Choice(["none", "z", "so3"]),
    default="none",
    help="Test-time voting rotation mode.",
    show_default=True,
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size for inference.",
    show_default=True,
)
def test(
    checkpoint: str,
    test_dir: str,
    output: str,
    num_votes: int,
    rotation_mode: str,
    batch_size: int,
):
    """Run inference with optional voting and save predictions."""
    from pointcls.test import run_test
    run_test(checkpoint, test_dir, output, num_votes, batch_size, rotation_mode=rotation_mode)


@cli.command()
@click.option(
    "--data-dir",
    default="data/modelnet40",
    help="Directory to download/extract ModelNet40.",
    show_default=True,
)
def train_all(data_dir: str):
    """Download data and train both DGCNN and PointMLP."""
    from pointcls.data.download import download_modelnet40, verify_modelnet40
    from pointcls.train import train_model

    print("=" * 60)
    print(" Pipeline: Train DGCNN + Train PointMLP")
    print("=" * 60)

    if not verify_modelnet40(data_dir):
        download_modelnet40(data_dir)

    # Train DGCNN
    dgcnn_config = "configs/dgcnn.yaml"
    print("\n>>> Training DGCNN...")
    dgcnn_best, dgcnn_class = train_model(dgcnn_config)

    # Train PointMLP
    pointmlp_config = "configs/pointmlp.yaml"
    print("\n>>> Training PointMLP...")
    pmlp_best, pmlp_class = train_model(pointmlp_config)

    # Print comparison
    print("\n" + "=" * 60)
    print(" COMPARISON")
    print("=" * 60)
    print(f"{'Model':<15} {'Inst Acc':>10} {'Class Acc':>10}")
    print("-" * 40)
    print(f"{'DGCNN':<15} {dgcnn_best:>10.4f} {dgcnn_class:>10.4f}")
    print(f"{'PointMLP':<15} {pmlp_best:>10.4f} {pmlp_class:>10.4f}")
    print("-" * 40)


@cli.command()
@click.option(
    "--dgcnn-checkpoint",
    required=True,
    help="Path to DGCNN checkpoint.",
)
@click.option(
    "--pointmlp-checkpoint",
    required=True,
    help="Path to PointMLP checkpoint.",
)
@click.option(
    "--test-dir",
    "-t",
    required=True,
    help="Directory containing test data.",
)
@click.option(
    "--output-dir",
    "-o",
    default="results",
    help="Directory for output CSVs.",
    show_default=True,
)
@click.option(
    "--num-votes",
    default=1,
    help="Number of votes per model.",
    show_default=True,
)
@click.option(
    "--rotation-mode",
    type=click.Choice(["none", "z", "so3"]),
    default="none",
    help="Test-time voting rotation mode.",
    show_default=True,
)
def test_both(
    dgcnn_checkpoint: str,
    pointmlp_checkpoint: str,
    test_dir: str,
    output_dir: str,
    num_votes: int,
    rotation_mode: str,
):
    """Test both DGCNN and PointMLP on the same test data."""
    import os
    from pointcls.test import run_test

    os.makedirs(output_dir, exist_ok=True)

    print(">>> DGCNN Inference...")
    run_test(
        dgcnn_checkpoint,
        test_dir,
        os.path.join(output_dir, "dgcnn_result.csv"),
        num_votes=num_votes,
        rotation_mode=rotation_mode,
    )

    print("\n>>> PointMLP Inference...")
    run_test(
        pointmlp_checkpoint,
        test_dir,
        os.path.join(output_dir, "pointmlp_result.csv"),
        num_votes=num_votes,
        rotation_mode=rotation_mode,
    )

    print(f"\nResults saved in: {output_dir}/")


if __name__ == "__main__":
    cli()
