"""Tests for webhook.py — the wan-dash completion webhook emit."""

from unittest.mock import patch

from config import TrainingJob
from webhook import _entry, _epoch, notify_webhook


def _job(webhook_url="https://hook.example/cb?job_id=abc", model_type="krea2", job_id="rp-job-1"):
    return TrainingJob(
        job_id=job_id,
        model_type=model_type,
        dataset_zip_url="x",
        trigger_word="kiannaugcstudio",
        webhook_url=webhook_url,
    )


def _upload_result(*epochs, variant=""):
    """Build a maybe_upload_outputs-shaped result with one entry per epoch."""
    files = []
    for ep in epochs:
        fn = f"kiannaugcstudio_krea2_epoch{ep}.safetensors"
        files.append({
            "filename": fn,
            "key": f"lora-outputs/rp-job-1/{fn}",
            "url": f"https://presigned/{fn}",
            "size": 1234,
            "noise_variant": variant,
        })
    return {"output_files": files, "presigned_urls": [f["url"] for f in files]}


class TestEpoch:
    def test_parses_epoch(self):
        assert _epoch("kiannaugcstudio_krea2_epoch100.safetensors") == 100

    def test_no_epoch_token(self):
        assert _epoch("kiannaugcstudio_krea2.safetensors") == 0

    def test_empty(self):
        assert _epoch("") == 0


class TestEntry:
    def test_phase_final_for_empty_variant(self):
        e = _entry({"filename": "x_krea2_epoch10.safetensors", "key": "k", "size": 9, "noise_variant": ""})
        assert e == {"filename": "x_krea2_epoch10.safetensors", "key": "k", "size": 9, "epoch": 10, "phase": "final"}

    def test_phase_passthrough(self):
        e = _entry({"filename": "x_wan2.2high_epoch5.safetensors", "key": "k", "size": 9, "noise_variant": "high"})
        assert e["phase"] == "high"


class TestNotifyWebhook:
    def test_noop_when_no_url(self):
        job = _job(webhook_url=None)
        with patch("webhook.requests.post") as post:
            notify_webhook(job, ok=True, upload_result=_upload_result(100), timing={"total_s": 10})
        post.assert_not_called()

    def test_success_payload_and_split(self):
        job = _job()
        with patch("webhook.requests.post") as post:
            notify_webhook(job, ok=True, upload_result=_upload_result(50, 100), timing={"total_s": 1000})
        post.assert_called_once()
        body = post.call_args.kwargs["json"]
        assert body["type"] == "train"
        assert body["status"] == "completed"
        assert body["model"] == "krea2"
        # max epoch (100) -> final lora; 50 -> checkpoint
        assert [f["epoch"] for f in body["lora_files"]] == [100]
        assert [f["epoch"] for f in body["checkpoints"]] == [50]
        assert body["lora_files"][0]["key"] == "lora-outputs/rp-job-1/kiannaugcstudio_krea2_epoch100.safetensors"
        assert body["lora_files"][0]["phase"] == "final"
        assert body["duration_sec"] == 1000
        assert body["cost_usd"] > 0

    def test_only_uploaded_entries(self):
        """Entries without a key (failed upload) are excluded from the payload."""
        job = _job()
        res = _upload_result(100)
        res["output_files"].append(
            {"filename": "x_krea2_epoch90.safetensors", "local_path": "/tmp/x", "upload_error": "boom"}
        )
        with patch("webhook.requests.post") as post:
            notify_webhook(job, ok=True, upload_result=res, timing={"total_s": 5})
        body = post.call_args.kwargs["json"]
        names = [f["filename"] for f in body["lora_files"] + body["checkpoints"]]
        assert "x_krea2_epoch90.safetensors" not in names

    def test_failure_payload(self):
        job = _job()
        with patch("webhook.requests.post") as post:
            notify_webhook(job, ok=False, error="OOM", timing={"total_s": 12})
        body = post.call_args.kwargs["json"]
        assert body == {"type": "train", "status": "failed", "model": "krea2", "error": "OOM"}

    def test_post_failure_swallowed(self):
        """A webhook POST exception must never raise (best-effort)."""
        job = _job()
        with patch("webhook.requests.post", side_effect=Exception("network")):
            notify_webhook(job, ok=False, error="x", timing={"total_s": 1})  # must not raise
