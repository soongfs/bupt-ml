"""ModelNet40 dataset download and extraction."""

import os
import ssl
import tarfile
import urllib.request
import sys

# HuggingFace Pointcept mirror. This tarball extracts directly into per-class
# directories containing CSV-style .txt point clouds.
HF_URL = (
    "https://huggingface.co/datasets/Pointcept/"
    "modelnet40_normal_resampled-compressed/resolve/main/"
    "modelnet40_normal_resampled.tar.gz"
)
DEFAULT_DATA_DIR = "data/modelnet40"

POINT_EXTENSIONS = (".off", ".txt")


def verify_modelnet40(data_dir: str) -> bool:
    if not os.path.isdir(data_dir):
        return False
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith("_")
        and not d.startswith(".")
        and d != "__MACOSX"
    ]
    if len(subdirs) != 40:
        return False
    return all(_class_has_point_files(os.path.join(data_dir, d)) for d in subdirs)


def download_modelnet40(data_dir: str = DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    if verify_modelnet40(data_dir):
        _remove_download_artifacts(data_dir)
        print(f"ModelNet40 already exists at {data_dir} (40 class directories found).")
        return

    archive_path = os.path.join(data_dir, "_dl_archive")
    partial_path = f"{archive_path}.part"

    # Download
    if os.path.exists(partial_path):
        os.remove(partial_path)

    if os.path.exists(archive_path) and not tarfile.is_tarfile(archive_path):
        print(f"Removing invalid archive artifact: {archive_path}")
        os.remove(archive_path)

    if not os.path.exists(archive_path):
        try:
            print(f"Downloading {HF_URL} ...")
            _download(HF_URL, partial_path)
            os.replace(partial_path, archive_path)
        except Exception as e:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            print(f"\nDownload failed: {e}")
            print("Please download the tar.gz manually and place it at:")
            print(f"  {archive_path}")
            return
    else:
        print(f"Archive already downloaded: {archive_path}")

    # Extract
    print(f"Extracting to {data_dir} ...")
    with tarfile.open(archive_path, "r:gz") as tf:
        _safe_extract(tf, data_dir)
    print("  Done.")

    # Clean up archive
    _remove_download_artifacts(data_dir)

    if verify_modelnet40(data_dir):
        print(f"ModelNet40 ready at {data_dir} (40 class directories).")
    else:
        print(f"WARNING: Expected 40 class directories but did not find them.")
        for item in sorted(os.listdir(data_dir)):
            print(f"  {item}")


def _download(url: str, dest: str):
    def _progress(count, block_size, total_size):
        if total_size <= 0:
            downloaded = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded {downloaded:.1f} MiB")
            sys.stdout.flush()
            return
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


def _class_has_point_files(class_dir: str) -> bool:
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(POINT_EXTENSIONS):
            return True

    for split in ("train", "test"):
        split_dir = os.path.join(class_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if fname.lower().endswith(POINT_EXTENSIONS):
                return True
    return False


def _safe_extract(tf: tarfile.TarFile, dest: str):
    dest_abs = os.path.abspath(dest)
    for member in tf.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if target != dest_abs and not target.startswith(dest_abs + os.sep):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tf.extractall(dest)


def _remove_download_artifacts(data_dir: str):
    for name in ("_dl_archive", "_dl_archive.part"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed {name}.")


if __name__ == "__main__":
    download_modelnet40()
