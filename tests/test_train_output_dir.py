import os
from pathlib import Path

import pytest
import torch

from pointcls.train import _check_resume_config_compatible, _resolve_output_dir


def test_resolve_output_dir_prefers_config_value():
    config = {"model": "pointnext", "output_dir": "runs/pointnext_2048"}

    assert _resolve_output_dir(config) == "runs/pointnext_2048"


def test_resolve_output_dir_defaults_to_model_name_for_backward_compatibility():
    config = {"model": "dgcnn"}

    assert _resolve_output_dir(config) == os.path.join("runs", "dgcnn")


def test_resume_config_guard_rejects_mismatched_num_points(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "config": {
                "model": "pointnext",
                "num_points": 1024,
                "use_normals": True,
                "rotation_mode": "z",
                "width": 64,
                "blocks": [1, 2, 2, 2],
                "strides": [1, 2, 2, 2],
                "nsample": [24, 24, 24, 24],
                "expansion": 4,
            }
        },
        checkpoint_path,
    )

    new_config = {
        "model": "pointnext",
        "num_points": 2048,
        "use_normals": True,
        "rotation_mode": "z",
        "width": 64,
        "blocks": [1, 2, 2, 2],
        "strides": [1, 2, 2, 2],
        "nsample": [32, 32, 24, 24],
        "expansion": 4,
    }

    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _check_resume_config_compatible(new_config, checkpoint_path)


def test_resume_config_guard_accepts_matching_key_config(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    config = {
        "model": "pointnext",
        "num_points": 2048,
        "use_normals": True,
        "rotation_mode": "z",
        "width": 64,
        "blocks": [1, 2, 2, 2],
        "strides": [1, 2, 2, 2],
        "nsample": [32, 32, 24, 24],
        "expansion": 4,
    }
    torch.save({"config": config}, checkpoint_path)

    _check_resume_config_compatible(dict(config), checkpoint_path)
