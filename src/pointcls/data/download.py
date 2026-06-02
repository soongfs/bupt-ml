"""ModelNet40 dataset download and extraction."""

import os
import shutil
import sys

DEFAULT_DATA_DIR = "data/modelnet40"
HF_REPO = "naderalfares/ModelNet40"


def verify_modelnet40(data_dir: str) -> bool:
    """Check that 40 class dirs exist, each with train/ and test/ subdirs."""
    if not os.path.isdir(data_dir):
        return False
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))
               and not d.startswith("_") and not d.startswith(".")]
    if len(subdirs) != 40:
        return False
    return all(
        os.path.isdir(os.path.join(data_dir, d, "train")) and
        os.path.isdir(os.path.join(data_dir, d, "test"))
        for d in subdirs
    )


def download_modelnet40(data_dir: str = DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 already exists at {data_dir}")
        return

    # Remove any previous Pointcept data
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        print("Removing old dataset...")
        shutil.rmtree(data_dir)
        os.makedirs(data_dir)

    print(f"Downloading {HF_REPO} from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=data_dir,
        )
    except ImportError:
        print("huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try the Pointcept fallback or Baidu Pan.")
        sys.exit(1)

    # Flatten: the repo has a top-level "ModelNet40/" directory
    nested = os.path.join(data_dir, "ModelNet40")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
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
