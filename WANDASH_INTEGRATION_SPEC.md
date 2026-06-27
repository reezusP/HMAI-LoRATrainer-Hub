# Spec: wan-dash integration readiness (fork-side only)

> Status: **DRAFT / not yet implemented.** This specs the changes to **this fork** so wan-dash
> can drive it like its current Modal trainer. It does NOT spec the wan-dash code changes —
> only the fork-side prep + the contract wan-dash will build against.

## Context

The AI-Toolkit trainer (this fork) runs on RunPod (endpoint `j15eua5j5mhrgv`) and produces good
**Krea2** / z_image LoRAs. wan-dash's training pipeline is **webhook-driven**
(`supabase/functions/training-webhook`): the current Modal trainer (`lora-training`) POSTs a
completion webhook and wan-dash stores the result. This fork's `handler.py` currently just returns
its result to RunPod — **no outbound callback** — so it can't slot into that flow.

**The only fork-side gap:** on completion, POST a webhook in wan-dash's existing shape.

**Enabler:** this fork already uploads LoRAs to `wan-outputs/lora-outputs/<job_id>/` — the *same R2
bucket* wan-dash uses — so the `key` we report is directly usable by wan-dash
(`getSignedUrl` / `copyR2Object`) with **no cross-account copy**.

Scope: fork-only · new AI-Toolkit-only models · **Krea2 primary** (z_image works too) ·
single-LoRA models (wan2.2 dual-noise deferred).

## Contract the fork must emit

POST to the caller-supplied `webhook_url` (wan-dash passes
`…/functions/v1/training-webhook?job_id=<wandashJobId>`; the fork needn't know that id — it's in the URL).

- **Success:** `{ "type":"train", "status":"completed", "lora_files":[…], "checkpoints":[…], "cost_usd":<float>, "duration_sec":<int> }`
- **Failure:** `{ "type":"train", "status":"failed", "error":<str> }`
- Each `lora_files` / `checkpoints` entry: `{ "filename", "key", "size", "epoch", "phase" }`
  - `key` = `lora-outputs/<runpod_job_id>/<filename>` (bucket `wan-outputs`).
  - `phase` = `"final"` for Krea2/z_image. (`"high"`/`"low"` reserved for future wan2.2.)
  - `epoch` parsed from filename `…_epoch<N>.safetensors` (`config.rename_output`).
  - `lora_files` = max-epoch (final) file; `checkpoints` = the rest. (Mirrors `training_app.py`.)

## Fork changes

1. **`config.py`** — add `webhook_url: str | None = None` to `TrainingJob`; accept optional
   `webhook_url` (string) in `validate_payload`; add `H100_USD_PER_SEC = 0.00116`.
   *(Optional)* add `"seed": ("train","seed")` to `overrides_translator._RENAME_SINKS` for seed parity.
2. **`uploader.py` → `maybe_upload_outputs`** — enrich each `output_files` entry with `key` (the
   `s3_key` it already builds) and `size`. No other behavior change.
3. **New `webhook.py`** — `notify_webhook(job, *, ok, upload_result, model_type, timing, error=None)`:
   no-op if `job.webhook_url` is falsy; parse epoch (`re.search(r"_epoch(\d+)", filename)`),
   `phase = noise_variant or "final"`; split max-epoch → `lora_files`, rest → `checkpoints`;
   `cost_usd = total_s * H100_USD_PER_SEC`, `duration_sec = round(total_s)`; `requests.post(…, timeout=30)`
   best-effort (try/except + log; never crash the job). `requests` already in `requirements.txt`.
4. **`handler.py`** — call `notify_webhook(...)` on success (after `maybe_upload_outputs`, before the
   final return) and on failure paths where `job` exists. Leave the RunPod return value intact so
   `/status` polling still works as a fallback.

Reuse existing helpers (`config.rename_output`, `uploader._detect_variant`, `uploader._get_s3_client`).

## What wan-dash will send (for reference — NOT built here)

```json
{ "input": {
  "model_type": "krea2",
  "dataset_zip_url": "<presigned wan-outputs zip of training-datasets/<jobId>/images/>",
  "trigger_word": "<trigger>",
  "config_overrides": { "epochs": 80, "save_every_n_epochs": 10,
                        "optimizer.lr": 1e-4, "adapter.rank": 32, "dataset.num_repeats": 1 },
  "webhook_url": "<supabase>/functions/v1/training-webhook?job_id=<wandashJobId>" } }
```
`config_overrides` keys are exactly this fork's `overrides_translator` allow-list. wan-dash-side work
(out of scope): add `krea2` to its model registry + cost table; zip the prefix; store the RunPod job id;
route AI-Toolkit jobs to the new RunPod endpoint.

## Verification

- **Unit (pytest, `tests/` style):** `tests/test_webhook.py` — epoch parse, `phase="final"`,
  max-epoch→`lora_files`/rest→`checkpoints`, failure shape. No real HTTP; assert the dict.
- **E2E:** push fork → RunPod rebuild (~7.5 min) → smoke `ok:true`; fire a Krea2 `/run` with a
  `webhook_url` capture URL; confirm POST body matches the contract and each `lora_files[].key`
  resolves under `wan-outputs/lora-outputs/…`.

## Out of scope (flagged dependency)

Serving the Krea2 LoRA cleanly needs a **diffusers Krea2 inference path** — today only z_image has one
(`z_image_diffusers_app.py`); ComfyUI mis-loads AI-Toolkit LoRAs. Separate follow-up.
