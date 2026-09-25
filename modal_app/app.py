"""Modal deployment of the Demucs worker, with a RunPod-style queue API.

Runs the same runpod/handler.py as the RunPod image, so inputs and outputs are identical.

Deploy from the repo root:
    modal deploy modal_app/app.py

HTTP API (Modal proxy auth: send Modal-Key / Modal-Secret headers from a proxy auth token):
    POST /run            {"input": {...}, "webhook"?: "https://..."}  -> {"id": "fc-...", "status": "IN_QUEUE"}
    POST /runsync        same body; waits up to 90 s, else returns {"id", "status": "IN_PROGRESS"}
    GET  /status/{id}                      -> {"id", "status", "output" | "error"}
    POST /cancel/{id}                      -> {"id", "status": "CANCELLED"}
status is IN_PROGRESS (queued or running), COMPLETED or FAILED.
With "webhook", the finished /status body is POSTed there, signed with WEBHOOK_SECRET from the
Modal secret "audio-webhook" (see _send_webhook). A crash or 600 s timeout sends nothing, so keep
polling /status as a fallback.

GCS upload uses the same env vars as RunPod, from Modal secrets: "demucs-gcs"
(GCS_SERVICE_ACCOUNT_JSON) and "demucs-config" (GCS_BUCKET, GCS_PREFIX). Without a bucket, stems
come back as base64.
"""

import time

import modal

MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_q", "mdx_extra", "mdx_extra_q"]
GPU = "L4"

app = modal.App("demucs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "build-essential")
    .pip_install("torch==2.7.1", "torchaudio==2.7.1")
    .pip_install_from_requirements("runpod/requirements.txt")
    # Empty DEMUCS_PRELOAD: handler.py must not touch the GPU at import; the model loads in @modal.enter.
    .env(
        {
            "TORCH_HOME": "/models/torch",
            "PYTHONPATH": "/opt/demucs:/opt/worker",
            "DEMUCS_PRELOAD": "",
        }
    )
    .add_local_dir("demucs", "/opt/demucs/demucs", copy=True, ignore=["**/__pycache__"])
    # Bake the weights (~2 GB) into the image so cold starts never download them.
    .run_commands(
        "python -c \"from demucs.pretrained import get_model; [get_model(n) for n in %r]\"" % MODELS
    )
    .add_local_file("runpod/handler.py", "/opt/worker/handler.py")
)

@app.cls(
    gpu=GPU,
    # Reserved, not a cap: ffmpeg decode and mp3 encode need real cores (stems encode in
    # parallel), and RAM grows with length: ~6 GB for a 4-min song, ~16 GB for a 40-min video.
    # Modal's default reservation is 0.125 core / 128 MiB.
    cpu=2.0,
    memory=16384,
    image=image,
    secrets=[
        modal.Secret.from_name("demucs-gcs"),  # GCS_SERVICE_ACCOUNT_JSON
        modal.Secret.from_name("demucs-config"),  # GCS_BUCKET, GCS_PREFIX
        modal.Secret.from_name("audio-webhook"),  # WEBHOOK_SECRET
    ],
    timeout=600,
    scaledown_window=30,
    max_containers=10,
    enable_memory_snapshot=True,
)
class Demucs:
    # Memory snapshot: imports and CPU model load are captured once, so new containers
    # restore them instead of redoing them. There's no GPU during this phase.
    @modal.enter(snap=True)
    def load_cpu(self):
        t0 = time.perf_counter()
        import handler
        from demucs.pretrained import get_model

        self.handler = handler
        self.model = get_model("htdemucs").eval()
        print(f"model loaded on cpu in {time.perf_counter() - t0:.2f}s")

    @modal.enter(snap=False)
    def load_gpu(self):
        t0 = time.perf_counter()
        import torch

        # handler.DEVICE was computed at import, before the GPU was attached.
        self.handler.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.handler._models["htdemucs"] = self.model.to(self.handler.DEVICE)
        print(f"model moved to {self.handler.DEVICE} in {time.perf_counter() - t0:.2f}s")

    @modal.method()
    def separate(self, job_id: str, inp: dict, submitted_at: float, webhook: str | None = None) -> dict:
        import resource

        import torch

        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            out = self.handler.handler({"id": job_id, "input": inp})
        except Exception as e:
            if webhook:
                _send_webhook(webhook, {"id": modal.current_function_call_id(), "status": "FAILED", "error": f"{type(e).__name__}: {e}"})
            raise
        out["queue_seconds"] = round(started - submitted_at, 3)
        out["total_seconds"] = round(time.time() - started, 3)
        out["peak_ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() // 2**20
        if webhook:
            _send_webhook(webhook, _status_body(modal.current_function_call_id(), out))
        return out


def _status_body(call_id: str, out) -> dict:
    """The /status response for a finished call; webhooks send the same body."""
    if isinstance(out, dict) and "error" in out:
        return {"id": call_id, "status": "FAILED", "error": out["error"]}
    return {"id": call_id, "status": "COMPLETED", "output": out}


def _send_webhook(url: str, body: dict) -> None:
    """POST body to url, signed with WEBHOOK_SECRET; retries 3x, never raises.

    Headers: X-Webhook-Timestamp: <unix seconds>
             X-Webhook-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{raw body}")>
    """
    import hashlib
    import hmac
    import json
    import os

    import requests

    raw = json.dumps(body, separators=(",", ":")).encode()
    for attempt, delay in enumerate((0, 2, 10)):
        time.sleep(delay)
        ts = str(int(time.time()))
        sig = hmac.new(os.environ["WEBHOOK_SECRET"].encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-Webhook-Timestamp": ts, "X-Webhook-Signature": f"v1={sig}"}
        try:
            r = requests.post(url, data=raw, headers=headers, timeout=10)
            if r.status_code < 300:
                return
            print(f"webhook attempt {attempt + 1}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"webhook attempt {attempt + 1}: {e}")
    print(f"webhook to {url} failed; the result is still available via /status")


web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")


@app.function(image=web_image)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    import uuid

    from fastapi import FastAPI, HTTPException

    web = FastAPI()

    async def spawn(body: dict):
        inp = body.get("input")
        if not isinstance(inp, dict):
            raise HTTPException(400, "body must be {\"input\": {...}, \"webhook\"?: \"https://...\"}")
        webhook = body.get("webhook")
        if webhook is not None and not (isinstance(webhook, str) and webhook.startswith("https://")):
            raise HTTPException(400, "webhook must be an https:// URL")
        return await Demucs().separate.spawn.aio(uuid.uuid4().hex, inp, time.time(), webhook)

    @web.post("/run")
    async def run(body: dict):
        call = await spawn(body)
        return {"id": call.object_id, "status": "IN_QUEUE"}

    @web.post("/runsync")
    async def runsync(body: dict):
        """Waits up to 90 s; if the job isn't done by then, returns IN_PROGRESS and the id to poll."""
        call = await spawn(body)
        try:
            out = await call.get.aio(timeout=90)
        except (TimeoutError, modal.exception.TimeoutError):
            return {"id": call.object_id, "status": "IN_PROGRESS"}
        except Exception as e:
            return {"id": call.object_id, "status": "FAILED", "error": f"{type(e).__name__}: {e}"}
        return _status_body(call.object_id, out)

    @web.get("/status/{call_id}")
    async def status(call_id: str):
        try:
            out = await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return {"id": call_id, "status": "IN_PROGRESS"}
        except modal.exception.NotFoundError:
            raise HTTPException(404, "job not found")
        except Exception as e:  # the call itself raised (crash, timeout, cancel)
            return {"id": call_id, "status": "FAILED", "error": f"{type(e).__name__}: {e}"}
        return _status_body(call_id, out)

    @web.post("/cancel/{call_id}")
    async def cancel(call_id: str):
        await modal.FunctionCall.from_id(call_id).cancel.aio()
        return {"id": call_id, "status": "CANCELLED"}

    return web
