"""ModelNet40 dataset download and extraction."""

import os
import shutil
import sys

DEFAULT_DATA_DIR = "data/modelnet40"
HF_REPO = "naderalfares/ModelNet40"


def verify_modelnet40(data_dir: str) -> bool:
    """Check that 40 class dirs exist with valid data files.

    Accepts two layouts:
    - Split: each class dir has train/ and test/ subdirectories.
    - Unsplit: each class dir contains .txt or .off files directly.
    """
    if not os.path.isdir(data_dir):
        return False
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))
               and not d.startswith("_") and not d.startswith(".")]
    if len(subdirs) != 40:
        return False

    # Split layout: class/train/ + class/test/
    split_ok = all(
        os.path.isdir(os.path.join(data_dir, d, "train")) and
        os.path.isdir(os.path.join(data_dir, d, "test"))
        for d in subdirs
    )
    if split_ok:
        return True

    # Unsplit layout: class/*.txt or class/*.off
    unsplit_ok = all(
        any(
            f.endswith((".txt", ".off"))
            for f in os.listdir(os.path.join(data_dir, d))
            if os.path.isfile(os.path.join(data_dir, d, f))
        )
        for d in subdirs
    )
    return unsplit_ok


def download_modelnet40(data_dir: str = DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 already exists at {data_dir}")
        return

    # Remove previous Pointcept data (txt format), keep partial OFF downloads
    if os.path.isdir(data_dir):
        has_off = any(
            f.endswith(".off") for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        )
        has_txt = any(
            f.endswith(".txt") for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        )
        if has_txt and not has_off:
            print("Removing old Pointcept dataset...")
            shutil.rmtree(data_dir)
            os.makedirs(data_dir)
        elif verify_modelnet40(data_dir):
            print(f"ModelNet40 already exists at {data_dir}")
            return

    print(f"Downloading {HF_REPO} from HuggingFace...")
    try:
        # Use hf-mirror.com in China for better speed/stability
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=data_dir,
            max_workers=1,
        )
    except ImportError:
        print("huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try the Pointcept fallback or Baidu Pan.")
        sys.exit(1)

    # Flatten: the repo has a top-level "ModelNet40/" directory.
    # Only move class subdirectories (those containing train/ and test/).
    nested = os.path.join(data_dir, "ModelNet40")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            if not os.path.isdir(src):
                continue
            if not (os.path.isdir(os.path.join(src, "train")) and
                    os.path.isdir(os.path.join(src, "test"))):
                continue  # Skip metadata dirs like "data/"
            dst = os.path.join(data_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(nested)
        print("  Flattened directory structure.")

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 ready at {data_dir} (40 classes, train/test splits).")
    else:
        print("WARNING: Dataset verification failed.")
        for item in sorted(os.listdir(data_dir)):
            print(f"  {item}")


if __name__ == "__main__":
    download_modelnet40()
