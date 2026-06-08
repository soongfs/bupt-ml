"""Download and verify the Pointcept ModelNet40 normal-resampled dataset."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_DATA_DIR = "data/modelnet40"
HF_REPO = "Pointcept/modelnet40_normal_resampled-compressed"
HF_ENDPOINT = "https://hf-mirror.com"
EXPECTED_NUM_CLASSES = 40
EXPECTED_NUM_SAMPLES = 9843
METADATA_FILES = {
    "filelist.txt",
    "modelnet40_shape_names.txt",
    "modelnet40_train.txt",
    "modelnet40_test.txt",
}


def _class_dirs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and not p.name.startswith("_")
        and p.name.lower() not in {"modelnet40", "__macosx"}
    )


def _is_modelnet40_root(path: str | Path) -> bool:
    root = Path(path)
    if not root.is_dir():
        return False
    classes = _class_dirs(root)
    if len(classes) != EXPECTED_NUM_CLASSES:
        return False
    return all(any(cls.glob("*.txt")) for cls in classes)


def verify_modelnet40(data_dir: str = DEFAULT_DATA_DIR) -> bool:
    """Return True only for the Pointcept unsplit 40-class TXT layout."""
    return _is_modelnet40_root(data_dir)


def _find_modelnet40_root(search_root: str | Path) -> Path | None:
    """Find the directory containing the 40 Pointcept class directories."""
    root = Path(search_root)
    candidates = [root]
    candidates.extend(p for p in root.rglob("*") if p.is_dir())
    for candidate in candidates:
        if _is_modelnet40_root(candidate):
            return candidate
    return None


def _clean_metadata_files(root: Path) -> None:
    for name in METADATA_FILES:
        path = root / name
        if path.exists() and path.is_file():
            path.unlink()


def _count_sample_txt(root: Path) -> int:
    return sum(len(list(cls.glob("*.txt"))) for cls in _class_dirs(root))


def _replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def download_modelnet40(data_dir: str = DEFAULT_DATA_DIR, force: bool = False) -> None:
    """Download Pointcept/modelnet40_normal_resampled-compressed from HuggingFace.

    The final directory is normalized to:
        data_dir/class_name/class_name_0001.txt

    No Baidu Pan, OFF mesh, or alternate dataset fallback is used.
    """
    target = Path(data_dir)
    if target.exists() and verify_modelnet40(str(target)) and not force:
        print(f"ModelNet40 already exists at {target}")
        return
    if target.exists() and force:
        shutil.rmtree(target)

    tmp_dir = target.parent / f".{target.name}_hf_download"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_REPO} from HuggingFace to {tmp_dir}...")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Install dependencies with: uv sync")
        sys.exit(1)

    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(tmp_dir),
            allow_patterns="*.txt",
            max_workers=1,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    found = _find_modelnet40_root(tmp_dir)
    if found is None:
        print("Dataset verification failed: could not find 40 class TXT directories.")
        print("Downloaded files:")
        for item in sorted(tmp_dir.rglob("*"))[:100]:
            print(f"  {item}")
        sys.exit(1)

    _clean_metadata_files(found)
    _replace_dir(found, target)
    if tmp_dir.exists() and tmp_dir != target:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not verify_modelnet40(str(target)):
        print("Dataset verification failed after normalization.")
        sys.exit(1)

    sample_count = _count_sample_txt(target)
    print(f"ModelNet40 ready at {target}")
    print(f"Classes: {len(_class_dirs(target))}; samples: {sample_count}")
    if sample_count != EXPECTED_NUM_SAMPLES:
        print(f"WARNING: expected {EXPECTED_NUM_SAMPLES} samples, got {sample_count}")


if __name__ == "__main__":
    download_modelnet40()
