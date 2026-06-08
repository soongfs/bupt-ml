from pathlib import Path

from pointcls.data.download import (
    DEFAULT_DATA_DIR,
    HF_REPO,
    _find_modelnet40_root,
    _is_modelnet40_root,
    verify_modelnet40,
)


def _write_sample(path: Path):
    path.write_text("0,0,0,1,0,0\n1,0,0,1,0,0\n")


def _make_minimal_classes(root: Path):
    for i in range(40):
        cls = f"class_{i:02d}"
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        _write_sample(class_dir / f"{cls}_0001.txt")


def test_download_constants_use_pointcept_hf_dataset():
    assert DEFAULT_DATA_DIR == "data/modelnet40"
    assert HF_REPO == "Pointcept/modelnet40_normal_resampled-compressed"


def test_verify_modelnet40_accepts_only_40_class_txt_layout(tmp_path: Path):
    _make_minimal_classes(tmp_path)

    assert verify_modelnet40(str(tmp_path))


def test_verify_modelnet40_rejects_official_split_or_off_files(tmp_path: Path):
    for i in range(40):
        cls = f"class_{i:02d}"
        train_dir = tmp_path / cls / "train"
        train_dir.mkdir(parents=True)
        (train_dir / f"{cls}_0001.off").write_text("OFF\n0 0 0\n")

    assert not verify_modelnet40(str(tmp_path))


def test_find_modelnet40_root_finds_nested_pointcept_layout(tmp_path: Path):
    nested = tmp_path / "snapshot" / "modelnet40_normal_resampled"
    _make_minimal_classes(nested)

    assert _find_modelnet40_root(tmp_path) == nested
    assert _is_modelnet40_root(nested)
