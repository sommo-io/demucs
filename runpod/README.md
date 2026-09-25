# RunPod serverless worker

Queue-based RunPod Serverless worker for Demucs music source separation (vocals, drums, bass, other;
guitar and piano with `htdemucs_6s`). All model weights are baked into the image. The input and output
match the Replicate cog, so `{"stem": "vocals"}` returns `vocals` + `no_vocals`.

Everything RunPod-specific lives in this directory (plus `.github/workflows/runpod-image.yml`).
Upstream files are untouched, so the fork keeps merging `facebookresearch/demucs` cleanly:

```bash
git remote add upstream https://github.com/facebookresearch/demucs.git
git pull upstream main
```

## Image

GitHub Actions builds `ghcr.io/sommo-io/demucs`:

- a `v*` git tag (e.g. `v1.0.0`) → `:v1.0.0`, `:v1.0`, `:v1`. Use these on the endpoint.
- every push to `main` that touches `demucs/` or `runpod/` → `:latest` and `:<commit sha>`

Release a new version:

```bash
git tag v1.0.1 && git push origin v1.0.1
```

Build locally from the repo root:

```bash
docker buildx build --platform linux/amd64 -f runpod/Dockerfile -t <image> --push .
```

The base image is a build arg (`BASE_IMAGE`, default `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`).
PyTorch 2.7 runs on Blackwell GPUs (RTX 50xx, RTX PRO 4500/6000 and their MIG slices), so every
pool works. The CUDA 12.8 image needs a 12.8+ host driver: set the endpoint's minimum CUDA version to 12.8.

## Deploy

RunPod console → Serverless → New Endpoint → queue-based, then either:

- **Docker Image**: `ghcr.io/sommo-io/demucs:v1.1.0`. The package must be public, or add GHCR registry
  credentials in RunPod (a GitHub token with `read:packages`). To roll out a release, tag it and point
  the endpoint at the new version.
- **GitHub Repo**: `sommo-io/demucs`, branch `main`, Dockerfile path `runpod/Dockerfile`, build context `.`
  (the repo root, since the Dockerfile copies `demucs/`). RunPod rebuilds and rolls out on every push.

GPU: 16 GB+ (A4000 / L4 / A5000 / 4090). Container disk: 20 GB. Idle timeout ~5 s.
For outputs as links instead of base64 (recommended for anything longer than a short clip), see [Output storage](#output-storage).

## Request

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/song.mp3", "stem": "vocals"}}'
```

| field | default | notes |
|---|---|---|
| `audio_url` / `audio_base64` | — | one is required; any format ffmpeg reads |
| `model` | `htdemucs` | `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `hdemucs_mmi`, `mdx`, `mdx_q`, `mdx_extra`, `mdx_extra_q` (`model_name` also works) |
| `stem` | — | return only this stem plus `no_<stem>`; omit for every stem the model has |
| `clip_mode` | `rescale` | `rescale` or `clamp` |
| `shifts` | `1` | 1–20; higher is slower but slightly better |
| `overlap` | `0.25` | overlap between the splits |
| `output_format` | `mp3` | `mp3`, `wav`, `flac` |
| `mp3_bitrate` | `320` | |
| `float32` | `false` | wav as float32 instead of 24-bit |
| `gcs_bucket` | `$GCS_BUCKET` | upload outputs to this GCS bucket |
| `gcs_prefix` | `$GCS_PREFIX` | object name prefix, e.g. `stems/` |

Response (one key per returned stem):

```json
{"model": "htdemucs", "format": "mp3", "sample_rate": 44100, "duration": 20.0, "inference_seconds": 1.2,
 "vocals": {"base64": "..."}, "no_vocals": {"base64": "..."}}
```

Only `htdemucs` is loaded at worker start; the other models load on first use and stay cached. Set
`DEMUCS_PRELOAD` (comma-separated, e.g. `htdemucs,htdemucs_ft`) to change that.

## Output storage

**Google Cloud Storage** (takes precedence). Set on the endpoint:

| env var | notes |
|---|---|
| `GCS_BUCKET` | default bucket; a request can override it with `gcs_bucket` |
| `GCS_PREFIX` | default object prefix; a request can override it with `gcs_prefix` |
| `GCS_SERVICE_ACCOUNT_JSON` | service account key, raw JSON or base64 of it. Needs `storage.objects.create` on the bucket (e.g. Storage Object Creator) |

Objects are named `<prefix><job id>-<stem>.<fmt>` (e.g. `-vocals.mp3`, `-no_vocals.mp3`), and each stem becomes
`{"gcs_uri": "gs://bucket/...", "url": "https://storage.googleapis.com/bucket/..."}`. The URL doesn't expire;
it opens for anyone only if the bucket grants `allUsers` Storage Object Viewer.

Credentials are read from env only, never from the request, so they don't end up in RunPod job logs.

**S3-compatible**: set `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`; each stem
becomes `{"url": "..."}`.

Decode the base64 output:

```bash
jq -r '.output.vocals.base64' resp.json | base64 -d > vocals.mp3
```

## Local test

```bash
docker run --gpus all --rm <image> python -u /app/handler.py --test_input "$(cat runpod/test_input.json)"
```

Falls back to CPU without a GPU (about 10 s for the 20 s test clip on an M-series Mac).
