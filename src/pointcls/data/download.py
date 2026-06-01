"""ModelNet40 dataset download and extraction."""

import os
import ssl
import tarfile
import urllib.request
import zipfile
import sys
import shutil

# Primary: HuggingFace (Pointcept mirror), Secondary: Stanford official
HF_URL = "https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed/resolve/main/modelnet40_normal_resampled.tar.gz"
STANFORD_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
DEFAULT_DATA_DIR = "data/modelnet40"

_URLS = [HF_URL, STANFORD_URL]


def verify_modelnet40(data_dir: str) -> bool:
    if not os.path.isdir(data_dir):
        return False
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    return len(subdirs) == 40


def download_modelnet40(data_dir: str = DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 already exists at {data_dir} (40 class directories found).")
        return

    archive_path = os.path.join(data_dir, "_dl_archive")
    archive_is_tar = False  # track format

    # Download
    if not os.path.exists(archive_path):
        last_error = None
        for url in _URLS:
            try:
                print(f"Downloading {url} ...")
                _download(url, archive_path)
                archive_is_tar = url.endswith(".tar.gz")
                break
            except Exception as e:
                last_error = e
                print(f"  Failed ({e}), trying next...")
        else:
            print(f"\nAll download sources failed. Last error: {last_error}")
            print(f"Please download manually from Baidu Pan:")
            print(f"  https://pan.baidu.com/s/1vHN3ECUT76NxFsJzne5RAQ  pwd: 2026")
            print(f"Place the zip at: {archive_path}")
            return
    else:
        print(f"Archive already downloaded: {archive_path}")
        # Auto-detect format: try tar.gz first, then zip
        try:
            with tarfile.open(archive_path, "r:gz") as tf:
                pass
            archive_is_tar = True
        except tarfile.ReadError:
            archive_is_tar = False

    # Extract
    print(f"Extracting to {data_dir} ...")
    if archive_is_tar:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(data_dir)
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(data_dir)
    print("  Done.")

    # Flatten: if extracted content is inside a single subdirectory,
    # move everything up one level
    items = os.listdir(data_dir)
    files_and_dirs = [i for i in items if os.path.isfile(os.path.join(data_dir, i)) or
                      os.path.isdir(os.path.join(data_dir, i))]
    nested = [i for i in items if os.path.isdir(os.path.join(data_dir, i))
              and i not in ("_dl_archive",)]

    # If everything is in one subdirectory, flatten
    if len(nested) == 1 and len(files_and_dirs) == len(nested) + (1 if os.path.exists(archive_path) else 0):
        inner = os.path.join(data_dir, nested[0])
        for item in os.listdir(inner):
            src = os.path.join(inner, item)
            dst = os.path.join(data_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(inner)
        print("  Flattened directory structure.")

    # Clean up archive
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print("  Removed archive file.")

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 ready at {data_dir} (40 class directories).")
    else:
        print(f"WARNING: Expected 40 class directories but did not find them.")
        for item in sorted(os.listdir(data_dir)):
            print(f"  {item}")


def _download(url: str, dest: str):
    def _progress(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / total_size))
        filled = int(40 * pct / 100)
        bar = "=" * filled + "-" * (40 - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%")
        sys.stdout.flush()

    for verify in (True, False):
        try:
            ctx = ssl.create_default_context()
            if not verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print("\n  Done.")
            return
        except (ssl.SSLError, urllib.error.URLError, OSError) as e:
            if verify:
                pass  # retry without verification
            else:
                raise


if __name__ == "__main__":
    download_modelnet40()
