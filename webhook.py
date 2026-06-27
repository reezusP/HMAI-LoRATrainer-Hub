"""Outbound completion webhook for the wan-dash integration.

When a TrainingJob carries a `webhook_url`, the handler calls `notify_webhook()` on
completion (success or failure). The payload matches what wan-dash's existing
`training-webhook` consumes, so this trainer can be a drop-in for the Modal trainer.

Best-effort: a webhook failure is logged and swallowed — it must never fail the job.
"""

from __future__ import annotations

import re

import requests
from loguru import logger

from config import H100_USD_PER_SEC, TrainingJob

_EPOCH_RE = re.compile(r"_epoch(\d+)")


def _epoch(filename: str) -> int:
    """Parse the epoch from an AI-Toolkit output filename (`..._epoch<N>.safetensors`)."""
    m = _EPOCH_RE.search(filename or "")
    return int(m.group(1)) if m else 0


def _entry(of: dict) -> dict:
    """Map an uploader `output_files` entry -> a wan-dash lora_files/checkpoints entry."""
    return {
        "filename": of["filename"],
        "key": of.get("key"),
        "size": of.get("size"),
        "epoch": _epoch(of["filename"]),
        "phase": of.get("noise_variant") or "final",
    }


def notify_webhook(
    job: TrainingJob,
    *,
    ok: bool,
    upload_result: dict | None = None,
    timing: dict | None = None,
    error: str | None = None,
) -> None:
    """POST a train-completion webhook to `job.webhook_url`. No-op if unset.

    Success → {type:"train", status:"completed", lora_files, checkpoints, cost_usd, duration_sec}
    Failure → {type:"train", status:"failed", error}
    `lora_files` is the max-epoch file(s); `checkpoints` is the rest.
    """
    webhook_url = getattr(job, "webhook_url", None)
    if not webhook_url:
        return

    total_s = float((timing or {}).get("total_s") or 0)

    try:
        if ok:
            # Only files that actually uploaded (have an R2 key).
            entries = [
                _entry(of)
                for of in (upload_result or {}).get("output_files", [])
                if of.get("key")
            ]
            max_epoch = max((e["epoch"] for e in entries), default=0)
            lora_files = [e for e in entries if e["epoch"] == max_epoch]
            checkpoints = [e for e in entries if e["epoch"] != max_epoch]
            payload = {
                "type": "train",
                "status": "completed",
                "model": job.model_type,
                "lora_files": lora_files,
                "checkpoints": checkpoints,
                "cost_usd": round(total_s * H100_USD_PER_SEC, 4),
                "duration_sec": round(total_s),
            }
        else:
            payload = {
                "type": "train",
                "status": "failed",
                "model": job.model_type,
                "error": error or "training failed",
            }

        requests.post(webhook_url, json=payload, timeout=30)
        logger.info(
            f"[webhook] posted {payload['status']} for job {job.job_id} "
            f"(lora={len(payload.get('lora_files', []))} ckpt={len(payload.get('checkpoints', []))})"
        )
    except Exception as e:  # best-effort: never fail the job over a webhook
        logger.warning(f"[webhook] post to {webhook_url} failed (non-fatal): {e}")
