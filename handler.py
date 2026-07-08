"""RunPod serverless entry point — thin orchestrator."""

import subprocess
import time
import traceback
from pathlib import Path

import runpod
from loguru import logger

from gpu import get_gpu_info

from config import (
    CLEANUP_DATASET_AFTER,
    PayloadError,
    TrainingResult,
    validate_payload,
)
from dataset import (
    count_dataset_media,
    download_dataset,
    download_dataset_prefix,
    extract_zip,
    validate_dataset,
)
from model_downloader import ensure_aria2c, ensure_model
from yaml_generator import generate_config
from trainer import run_training
from uploader import maybe_upload_outputs
from webhook import notify_webhook

CIVITAI_DOWNLOADER_DIR = Path("/app/CivitAI_Downloader")
CIVITAI_DOWNLOADER_REPO = "https://github.com/Hearmeman24/CivitAI_Downloader.git"


def _create_workspace(job):
    """Create the job workspace directories."""
    for d in [job.job_dir, job.dataset_dir, job.configs_dir, job.output_dir, job.logs_dir]:
        d.mkdir(parents=True, exist_ok=True)


def _ensure_free_space(job):
    """Reclaim stale space on the network volume before a run.

    The volume caches models + keeps per-job output dirs, so multi-model eval runs
    can fill it and cause `[Errno 122] Disk quota exceeded` mid-training. Before each
    run we free space by deleting:
      1. prior job dirs (>1h old) — never the current job, never a concurrently
         running sibling (age guard); their outputs are already uploaded to R2.
      2. cached models for OTHER model_types — the CURRENT model's dirs are always
         spared, so we never re-pull the (huge) Wan repo; other models re-download
         on demand next time they're used.
    Best-effort + logged; never fatal.
    """
    import os
    import shutil
    import time as _time

    from config import JOBS_DIR, MODELS_DIR

    def _dir_size(p):
        total = 0
        for root, _dirs, files in os.walk(p):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def _gb(b):
        return round(b / (1024 ** 3), 1)

    try:
        spare = set()
        for item in (getattr(job.model_spec, "downloads", None) or []):
            sub = getattr(item, "local_subdir", None)
            if sub:
                spare.add(sub)

        total, _used, free = shutil.disk_usage(str(MODELS_DIR.parent))
        logger.info(
            f"[freespace] volume total={_gb(total)}GB free={_gb(free)}GB; "
            f"sparing current-model dirs {sorted(spare)}"
        )

        freed = 0

        # 1) stale prior job dirs (spare current + any <1h-old concurrent sibling)
        now = _time.time()
        if JOBS_DIR.exists():
            for d in JOBS_DIR.iterdir():
                if not d.is_dir() or d.name == job.job_id:
                    continue
                try:
                    if now - d.stat().st_mtime < 3600:
                        continue
                    sz = _dir_size(d)
                    shutil.rmtree(d, ignore_errors=True)
                    freed += sz
                    logger.info(f"[freespace] removed stale job dir {d.name} (~{_gb(sz)}GB)")
                except OSError as e:
                    logger.warning(f"[freespace] job dir {d}: {e}")

        # 2) other models' caches (current model spared -> no big re-download)
        if MODELS_DIR.exists():
            for d in MODELS_DIR.iterdir():
                if not d.is_dir() or d.name in spare:
                    continue
                try:
                    sz = _dir_size(d)
                    shutil.rmtree(d, ignore_errors=True)
                    freed += sz
                    logger.info(f"[freespace] removed non-current model cache {d.name} (~{_gb(sz)}GB)")
                except OSError as e:
                    logger.warning(f"[freespace] model dir {d}: {e}")

        _t2, _u2, free_after = shutil.disk_usage(str(MODELS_DIR.parent))
        logger.info(f"[freespace] freed ~{_gb(freed)}GB; free now {_gb(free_after)}GB")
    except Exception as e:  # never let cleanup break a run
        logger.warning(f"[freespace] step failed (non-fatal): {e}")


def _cleanup_dataset(job):
    """Remove dataset files to free space after training."""
    import shutil
    try:
        zip_path = job.job_dir / "dataset.zip"
        if zip_path.exists():
            zip_path.unlink()
        if job.dataset_dir.exists():
            shutil.rmtree(job.dataset_dir)
        logger.info("Cleaned up dataset files")
    except Exception as e:
        logger.warning(f"Dataset cleanup failed: {e}")


def _ensure_civitai_deps():
    """Install aria2c and requests if not already available."""
    ensure_aria2c()

    try:
        import requests  # noqa: F401
    except ImportError:
        logger.info("Installing requests...")
        result = subprocess.run(
            ["pip", "install", "-q", "requests"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install requests: {result.stderr}")
        logger.info("requests installed")


def _maybe_download_civitai(job):
    """Download a CivitAI model if civitai_model_id is set. Returns True if downloaded."""
    if not job.civitai_model_id:
        return False

    logger.info(f"CivitAI model requested: {job.civitai_model_id}")

    _ensure_civitai_deps()

    # Clone the downloader tool if not present
    if not CIVITAI_DOWNLOADER_DIR.exists():
        logger.info("Cloning CivitAI_Downloader...")
        clone_result = subprocess.run(
            ["git", "clone", CIVITAI_DOWNLOADER_REPO, str(CIVITAI_DOWNLOADER_DIR)],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone CivitAI_Downloader: {clone_result.stderr or clone_result.stdout}"
            )

    # Download to a per-job directory (cleaned up with the job)
    download_dir = job.job_dir / "civitai_model"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading CivitAI model {job.civitai_model_id} to {download_dir}")
    result = subprocess.run(
        [
            "python", str(CIVITAI_DOWNLOADER_DIR / "download_with_aria.py"),
            "-m", job.civitai_model_id,
            "-o", str(download_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"CivitAI download failed (exit {result.returncode})")
        if result.stdout:
            logger.error(f"stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(
            f"CivitAI download failed for model {job.civitai_model_id}: "
            f"{result.stderr or result.stdout or 'unknown error'}"
        )

    # Find the .safetensors file
    safetensors_files = list(download_dir.glob("**/*.safetensors"))
    if not safetensors_files:
        raise RuntimeError(
            f"CivitAI download for model {job.civitai_model_id} produced no .safetensors file"
        )

    checkpoint_path = str(safetensors_files[0])
    job.civitai_checkpoint_path = checkpoint_path
    logger.info(f"CivitAI checkpoint ready: {checkpoint_path}")
    return True


def handler(event):
    """Main RunPod handler."""
    return _handler_inner(event)


def _handler_inner(event):
    timing = {}
    t_start = time.time()
    raw_input = event.get("input", {})
    raw_input["job_id"] = event.get("id", "local")
    job = None

    try:
        # --- Log raw input + GPU info ---
        gpu_info = get_gpu_info()
        logger.info(f"Raw input: {raw_input}")
        logger.info(f"GPUs: {gpu_info.count}x {gpu_info.name} ({gpu_info.vram_gb}GB each)")

        # --- Validate ---
        job = validate_payload(raw_input)
        logger.info(f"Job {job.job_id}: model={job.model_type}, trigger={job.trigger_word}")
        if job.config_overrides:
            logger.info(f"Config overrides: {job.config_overrides}")

        # --- Smoke test: validate + resolve only, no download/train ---
        if job.smoke:
            logger.info("Smoke test: payload valid, model resolved — skipping training.")
            timing["total_s"] = round(time.time() - t_start, 1)
            return {
                "ok": True,
                "smoke": True,
                "model_type": job.model_type,
                "trigger_word": job.trigger_word,
                "model": job.model_spec.name_or_path,
                "timing": timing,
            }

        # --- Workspace ---
        _create_workspace(job)

        # --- Free stale volume space (prevents EDQUOT mid-run) ---
        _ensure_free_space(job)

        # --- Dataset ---
        t0 = time.time()
        if job.dataset_r2_prefix:
            download_dataset_prefix(job.dataset_r2_prefix, job.dataset_dir)
        else:
            zip_path = job.job_dir / "dataset.zip"
            download_dataset(job.dataset_zip_url, zip_path)
            extract_zip(zip_path, job.dataset_dir)
        unmatched = validate_dataset(job.dataset_dir)
        if unmatched:
            logger.warning(f"Unmatched files (no captions): {unmatched}")
        timing["dataset_download_s"] = round(time.time() - t0, 1)

        # Count media files for the config generator (step/save_every math).
        img_count = count_dataset_media(job.dataset_dir)
        logger.info(f"Dataset media count: {img_count}")

        # --- Model ---
        t0 = time.time()
        if not _maybe_download_civitai(job):
            ensure_model(job)
        timing["model_download_s"] = round(time.time() - t0, 1)

        # --- Config ---
        generate_config(job, img_count)

        # --- Train ---
        t0 = time.time()
        logger.info(f"Training started: {job.model_type}")
        result = run_training(job)
        timing["training_s"] = round(time.time() - t0, 1)

        if not result.ok:
            timing["total_s"] = round(time.time() - t_start, 1)
            logger.error(f"Training failed: {result.error}")
            notify_webhook(job, ok=False, error=result.error, timing=timing)
            return {
                "ok": False,
                "error": result.error,
                "error_type": result.error_type or "TRAINING",
                "timing": timing,
            }

        # --- Upload ---
        t0 = time.time()
        upload_result = maybe_upload_outputs(job)
        timing["upload_s"] = round(time.time() - t0, 1)
        timing["total_s"] = round(time.time() - t_start, 1)
        logger.info(
            f"Job complete: {len(upload_result.get('output_files', []))} files, "
            f"total={timing['total_s']}s"
        )

        # --- Notify wan-dash (no-op if no webhook_url) ---
        notify_webhook(job, ok=True, upload_result=upload_result, timing=timing)

        # --- Cleanup ---
        if CLEANUP_DATASET_AFTER:
            _cleanup_dataset(job)

        return {
            "ok": True,
            "model_type": job.model_type,
            "trigger_word": job.trigger_word,
            "output_files": upload_result.get("output_files", []),
            "presigned_urls": upload_result.get("presigned_urls", []),
            "timing": timing,
        }

    except PayloadError as e:
        logger.error(f"Payload validation error: {e}")
        return {"ok": False, "error": str(e), "error_type": "VALIDATION"}
    except Exception as e:
        logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
        timing["total_s"] = round(time.time() - t_start, 1)
        if job is not None:
            notify_webhook(job, ok=False, error=str(e), timing=timing)
        return {
            "ok": False,
            "error": str(e),
            "error_type": "UNKNOWN",
            "timing": timing,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
