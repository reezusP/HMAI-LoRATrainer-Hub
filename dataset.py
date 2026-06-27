"""Dataset download, extraction, and validation."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import httpx
from loguru import logger

from config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 256 * 1024  # 256KB


def download_dataset(url: str, dest_path: Path) -> None:
    """Download a file from url to dest_path with streaming."""
    logger.info(f"Downloading dataset from {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (CHUNK_SIZE * 40) < CHUNK_SIZE:
                        pct = downloaded / total * 100
                        logger.info(f"Download progress: {pct:.0f}%")

    logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB to {dest_path}")


def download_dataset_prefix(prefix: str, dest_dir: Path) -> None:
    """Download every object under an R2 prefix into dest_dir (flat).

    Alternative to a zip URL — mirrors the Modal trainer's r2_download_dataset:
    list_objects_v2 + download_file. Reuses the uploader's configured R2/S3 client.
    """
    from uploader import _get_s3_client  # reuse the same R2/S3 config

    s3, bucket, _ = _get_s3_client()
    if s3 is None:
        raise ValueError(
            "dataset_r2_prefix given but no R2/S3 credentials configured"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    norm_prefix = prefix.rstrip("/") + "/"
    logger.info(f"Downloading dataset from r2://{bucket}/{norm_prefix}")

    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=norm_prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].split("/")[-1]
            if not name:  # skip "directory" placeholder keys
                continue
            s3.download_file(bucket, obj["Key"], str(dest_dir / name))
            count += 1

    if count == 0:
        raise ValueError(f"No objects found under R2 prefix '{norm_prefix}'")
    logger.info(f"Downloaded {count} files from r2://{bucket}/{norm_prefix}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file, handling nested single-folder zips and skipping __MACOSX."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if not m.startswith("__MACOSX") and not m.startswith("._")
        ]
        for member in members:
            zf.extract(member, dest_dir)

    # If everything extracted into a single subfolder, unwrap it
    children = [c for c in dest_dir.iterdir() if not c.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        nested_dir = children[0]
        logger.info(f"Unwrapping nested folder: {nested_dir.name}")
        for item in nested_dir.iterdir():
            target = dest_dir / item.name
            shutil.move(str(item), str(target))
        nested_dir.rmdir()


def validate_dataset(dataset_dir: Path) -> list[str]:
    """Validate that all media files have matching caption .txt files.

    Returns a list of media filenames that are missing captions.
    Returns empty list if all files have captions.
    Raises ValueError if no media files found.
    """
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    media_files = [
        f for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in all_extensions
    ]

    if not media_files:
        raise ValueError(f"No media files found in {dataset_dir}")

    unmatched = []
    for media_file in media_files:
        caption_file = media_file.with_suffix(".txt")
        if not caption_file.exists():
            unmatched.append(media_file.name)

    logger.info(
        f"Dataset: {len(media_files)} media files, "
        f"{len(media_files) - len(unmatched)} with captions"
    )
    return unmatched


def count_dataset_media(dataset_dir: Path) -> int:
    """Count the number of media files in dataset_dir.

    Returns 0 if the directory does not exist. The count drives the
    epoch→step math in the YAML generator (steps = epochs * img_count /
    batch_size * gradient_accumulation).
    """
    if not dataset_dir.exists():
        return 0
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sum(
        1 for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in all_extensions
    )
