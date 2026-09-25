"""RunPod serverless handler for Demucs music source separation.

Input (job["input"]):
    audio_url        str   URL of the source audio (any format ffmpeg can read), or
    audio_base64     str   base64-encoded audio file bytes
    model            str   "htdemucs" (default) | "htdemucs_ft" | "htdemucs_6s" | "hdemucs_mmi" |
                           "mdx" | "mdx_q" | "mdx_extra" | "mdx_extra_q"  (alias: model_name)
    stem             str   two-stem mode: return only this stem and "no_<stem>" (the rest mixed),
                           e.g. "vocals" -> vocals + no_vocals. Omit to return every stem.
    clip_mode        str   "rescale" (default) | "clamp"
    shifts           int   random shifts for equivariant stabilization, 1-20 (default 1)
    overlap          float overlap between the splits, 0-0.99 (default 0.25)
    output_format    str   "mp3" (default) | "wav" | "flac"
    mp3_bitrate      int   default 320
    float32          bool  save wav as float32 (default false); otherwise 24-bit
    gcs_bucket       str   upload outputs to this GCS bucket (overrides GCS_BUCKET env)
    gcs_prefix       str   object name prefix, e.g. "stems/2026/" (overrides GCS_PREFIX env)

Output:
    {"model": ..., "format": ..., "sample_rate": 44100, "duration": s, "inference_seconds": s,
     "vocals": <audio>, "no_vocals": <audio>, ...}   one key per returned stem
    <audio> is, in order of precedence:
      {"gcs_uri": "gs://...", "url": "https://..."}  when a GCS bucket is set (input or GCS_BUCKET)
      {"url": ...}                                   when S3 env vars are set (BUCKET_ENDPOINT_URL, ...)
      {"base64": ...}                                otherwise

GCS env vars:
    GCS_BUCKET                default bucket
    GCS_PREFIX                default object name prefix
    GCS_SERVICE_ACCOUNT_JSON  service account key, raw JSON or base64 of it; if unset,
                              Application Default Credentials are used

"url" is the plain https://storage.googleapis.com/<bucket>/<object> link; it only opens
if the bucket allows reads (public, or the caller has access).
"""

import base64
import binascii
import json
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import runpod
import torch
import torchaudio
from google.cloud import storage
from google.oauth2 import service_account
from runpod.serverless.utils import rp_upload

from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.pretrained import get_model

MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_q",
    "mdx_extra",
    "mdx_extra_q",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORMATS = ("mp3", "wav", "flac")
CONTENT_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}

_models = {}


def _model(name: str):
    if name not in _models:
        model = get_model(name)
        model.to(DEVICE).eval()
        _models[name] = model
    return _models[name]


# Load the default model(s) once per worker so only the first cold start pays for it;
# the others load on first use and stay cached.
for _name in filter(None, os.environ.get("DEMUCS_PRELOAD", "htdemucs").split(",")):
    _model(_name.strip())

_gcs_client = None


def _gcs() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        raw = os.environ.get("GCS_SERVICE_ACCOUNT_JSON", "").strip()
        if raw:
            if not raw.startswith("{"):
                raw = base64.b64decode(raw).decode("utf-8")
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info)
            _gcs_client = storage.Client(project=info.get("project_id"), credentials=creds)
        else:
            _gcs_client = storage.Client()
    return _gcs_client


def _upload_gcs(data: bytes, bucket: str, name: str, fmt: str) -> dict:
    blob = _gcs().bucket(bucket).blob(name)
    blob.upload_from_string(data, content_type=CONTENT_TYPES[fmt])
    return {"gcs_uri": f"gs://{bucket}/{name}", "url": blob.public_url}


def _fetch_audio(inp: dict) -> str:
    """Write the input audio to a temp file and return its path."""
    if inp.get("audio_url"):
        url = inp["audio_url"]
        suffix = os.path.splitext(url.split("?")[0])[1] or ".audio"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.content
    elif inp.get("audio_base64"):
        b64 = inp["audio_base64"]
        if "," in b64[:100]:  # strip data: URI prefix
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        suffix = ".audio"
    else:
        raise ValueError("Provide 'audio_url' or 'audio_base64'")

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _load_track(path: str, channels: int, samplerate: int) -> torch.Tensor:
    """demucs.separate.load_track, but raising instead of calling sys.exit()."""
    try:
        return AudioFile(path).read(streams=0, samplerate=samplerate, channels=channels)
    except (FileNotFoundError, subprocess.CalledProcessError) as ffmpeg_err:
        try:
            wav, sr = torchaudio.load(path)
        except Exception as ta_err:
            raise ValueError(f"ffmpeg: {ffmpeg_err}; torchaudio: {ta_err}") from ta_err
        return convert_audio(wav, sr, samplerate, channels)


def _encode(wav: torch.Tensor, fmt: str, job_id: str, name: str, gcs: tuple, save_kwargs: dict) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"{name}.{fmt}")
        save_audio(wav, path, **save_kwargs)

        bucket, prefix = gcs
        if bucket:
            with open(path, "rb") as f:
                data = f.read()
            try:
                return _upload_gcs(data, bucket, f"{prefix}{job_id}-{name}.{fmt}", fmt)
            except Exception as e:
                raise RuntimeError(f"GCS upload to gs://{bucket} failed: {e}") from e
        if os.environ.get("BUCKET_ENDPOINT_URL"):
            url = rp_upload.upload_file_to_bucket(f"{job_id}-{name}-{uuid.uuid4().hex[:8]}.{fmt}", path)
            return {"url": url}
        with open(path, "rb") as f:
            return {"base64": base64.b64encode(f.read()).decode("ascii")}


def handler(job):
    inp = job.get("input") or {}
    job_id = job.get("id", "local")

    model_name = inp.get("model") or inp.get("model_name") or "htdemucs"
    stem = inp.get("stem") or None
    clip_mode = inp.get("clip_mode", "rescale").lower()
    shifts = int(inp.get("shifts", 1))
    overlap = float(inp.get("overlap", 0.25))
    fmt = inp.get("output_format", "mp3").lower()
    gcs = (
        inp.get("gcs_bucket") or os.environ.get("GCS_BUCKET"),
        inp.get("gcs_prefix", os.environ.get("GCS_PREFIX", "")),
    )

    if model_name not in MODELS:
        return {"error": f"model must be one of {MODELS}, got {model_name!r}"}
    if fmt not in FORMATS:
        return {"error": f"output_format must be one of {list(FORMATS)}, got {fmt!r}"}
    if clip_mode not in ("rescale", "clamp"):
        return {"error": f"clip_mode must be rescale|clamp, got {clip_mode!r}"}
    if not 1 <= shifts <= 20:
        return {"error": f"shifts must be 1-20, got {shifts}"}
    if not 0 <= overlap < 1:
        return {"error": f"overlap must be in [0, 1), got {overlap}"}

    model = _model(model_name)
    if stem is not None and stem not in model.sources:
        return {"error": f"stem {stem!r} is not in {model_name}; supported: {model.sources}"}

    try:
        path = _fetch_audio(inp)
    except (requests.RequestException, ValueError, binascii.Error) as e:
        return {"error": f"Could not read input audio: {e}"}

    try:
        wav = _load_track(path, model.audio_channels, model.samplerate)
    except ValueError as e:
        return {"error": f"Could not decode input audio: {e}"}
    finally:
        os.remove(path)

    t0 = time.perf_counter()
    out = {
        "model": model_name,
        "format": fmt,
        "sample_rate": model.samplerate,
        "duration": round(wav.shape[-1] / model.samplerate, 3),
    }

    try:
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8  # epsilon keeps silent input from turning into NaN
        with torch.no_grad():
            sources = apply_model(
                model, ((wav - mean) / std)[None], device=DEVICE, shifts=shifts, split=True, overlap=overlap
            )[0]
        sources = (sources * std + mean).cpu()
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if stem is None:
        stems = dict(zip(model.sources, sources))
    else:
        i = model.sources.index(stem)
        stems = {stem: sources[i], f"no_{stem}": sources.sum(0) - sources[i]}

    save_kwargs = {
        "samplerate": model.samplerate,
        "bitrate": int(inp.get("mp3_bitrate", 320)),
        "clip": clip_mode,
        "as_float": bool(inp.get("float32", False)),
        "bits_per_sample": 24,
    }
    # Encode stems in parallel: mp3 encoding is CPU-bound, releases the GIL, and the GPU sits idle meanwhile.
    with ThreadPoolExecutor(len(stems)) as pool:
        futures = {name: pool.submit(_encode, source, fmt, job_id, name, gcs, save_kwargs) for name, source in stems.items()}
        for name, fut in futures.items():
            out[name] = fut.result()

    out["inference_seconds"] = round(time.perf_counter() - t0, 3)
    return out


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
