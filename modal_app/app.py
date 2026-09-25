"""Modal deployment of the Demucs worker, with a RunPod-style queue API.

Runs the same runpod/handler.py as the RunPod image, so inputs and outputs are identical.

Deploy from the repo root:
    modal deploy modal_app/app.py

HTTP API (Modal proxy auth: send Modal-Key / Modal-Secret headers from a proxy auth token):
    POST /run            {"input": {...}, "webhook"?: "https://..."}  -> {"id": "fc-...", "status": "IN_QUEUE"}
    POST /runsync        same body; waits up to 90 s, else returns {"id", "status": "IN_PROGRESS"}
    GET  /status/{id}                      -> {"id", "status", "output" | "error"}
    POST /cancel/{id}                      -> {"id", "status": "CANCELLED"}
status is IN_QUEUE, IN_PROGRESS (a worker picked it up), COMPLETED, FAILED, CANCELLED or TIMED_OUT.
With "webhook", the final /status body is POSTed there for every terminal status (COMPLETED,
FAILED, CANCELLED, TIMED_OUT), signed with WEBHOOK_SECRET from the Modal secret "audio-webhook";
delivery is done by the `watch` function and retried for ~12 min (see _send_webhook).

GCS upload uses the same env vars as RunPod, from Modal secrets: "demucs-gcs"
(GCS_SERVICE_ACCOUNT_JSON) and "demucs-config" (GCS_BUCKET, GCS_PREFIX). Without a bucket, stems
come back as base64.
"""

import time

import modal

MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_q", "mdx_extra", "mdx_extra_q"]
GPU = "L4"

app = modal.App("demucs")

# call id -> start time, written when a worker picks the job up, so /status can tell
# IN_QUEUE from IN_PROGRESS (Modal itself only knows "not finished yet").
started = modal.Dict.from_name("demucs-started", create_if_missing=True)

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
    timeout=900,  # 15 min per job; a 40-min video takes ~4 min
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
        # One tiny CPU pass so every lazy import happens now and lands in the snapshot. Otherwise
        # the first request imports torch._dynamo (via einops); a cancel mid-import leaves it
        # half-initialized and every later request in that container fails.
        import torch
        import torch._dynamo  # noqa: F401
        from demucs.apply import apply_model

        with torch.no_grad():
            apply_model(self.model, torch.zeros(1, 2, self.model.samplerate), device="cpu", split=True)
        import lameenc  # noqa: F401  (demucs.audio.encode_mp3 imports it lazily)

        handler._gcs()  # google-cloud-storage import + client construction
        print(f"model loaded and warmed on cpu in {time.perf_counter() - t0:.2f}s")

    @modal.enter(snap=False)
    def load_gpu(self):
        t0 = time.perf_counter()
        import torch

        # handler.DEVICE was computed at import, before the GPU was attached.
        self.handler.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.handler._models["htdemucs"] = self.model.to(self.handler.DEVICE)
        print(f"model moved to {self.handler.DEVICE} in {time.perf_counter() - t0:.2f}s")

    @modal.method()
    def separate(self, job_id: str, inp: dict, submitted_at: float) -> dict:
        import resource

        import torch

        started_at = time.time()
        started[modal.current_function_call_id()] = started_at
        torch.cuda.reset_peak_memory_stats()
        try:
            out = self.handler.handler({"id": job_id, "input": inp})
        except (AttributeError, ImportError) as e:
            if "partially initialized module" in str(e) or isinstance(e, ImportError):
                # Module state is corrupt (e.g. an import interrupted by a cancel): stop taking
                # inputs so Modal replaces this container instead of failing every job on it.
                modal.experimental.stop_fetching_inputs()
            raise
        out["queue_seconds"] = round(started_at - submitted_at, 3)
        out["total_seconds"] = round(time.time() - started_at, 3)
        out["peak_ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() // 2**20
        return out


def _status_body(call_id: str, out) -> dict:
    """The /status response for a finished call; webhooks send the same body."""
    if isinstance(out, dict) and "error" in out:
        return {"id": call_id, "status": "FAILED", "error": out["error"]}
    return {"id": call_id, "status": "COMPLETED", "output": out}


def _error_status(call_id: str, e: Exception) -> dict:
    """Map an exception from FunctionCall.get to a RunPod-style terminal status."""
    if isinstance(e, modal.exception.FunctionTimeoutError):
        status = "TIMED_OUT"
    elif "cancelled" in str(e).lower():  # a cancelled call surfaces as RemoteError("Function call was cancelled ...")
        status = "CANCELLED"
    else:
        status = "FAILED"
    return {"id": call_id, "status": status, "error": f"{type(e).__name__}: {e}"}


def _send_webhook(url: str, body: dict) -> None:
    """POST body to url, signed with WEBHOOK_SECRET; 5 attempts over ~12 min, never raises.

    Headers: X-Webhook-Timestamp: <unix seconds>
             X-Webhook-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{raw body}")>
    """
    import hashlib
    import hmac
    import json
    import os

    import requests

    raw = json.dumps(body, separators=(",", ":")).encode()
    for attempt, delay in enumerate((0, 5, 30, 120, 600)):
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


web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "requests")


@app.function(image=web_image, secrets=[modal.Secret.from_name("audio-webhook")], timeout=6 * 3600)
@modal.concurrent(max_inputs=1000)
async def watch(call_id: str, webhook: str) -> None:
    """Wait for a job to reach ANY terminal state and deliver the webhook.

    Runs outside the GPU worker, so timeouts, crashes and cancellations still produce a
    webhook (TIMED_OUT / FAILED / CANCELLED), like RunPod's platform-side webhooks.
    Only waits, so one small CPU container serves many jobs at once.
    """
    import asyncio

    try:
        body = _status_body(call_id, await modal.FunctionCall.from_id(call_id).get.aio())
    except Exception as e:
        body = _error_status(call_id, e)
    await asyncio.to_thread(_send_webhook, webhook, body)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)  # requests only enqueue/poll/wait, so one container serves many
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
        call = await Demucs().separate.spawn.aio(uuid.uuid4().hex, inp, time.time())
        if webhook:
            await watch.spawn.aio(call.object_id, webhook)
        return call

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
        except TimeoutError:  # builtin: not finished within 90 s (a job timeout is FunctionTimeoutError)
            running = await started.contains.aio(call.object_id)
            return {"id": call.object_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except Exception as e:
            return _error_status(call.object_id, e)
        return _status_body(call.object_id, out)

    def check_id(call_id: str) -> None:
        if not call_id.startswith("fc-"):  # Modal raises ExecutionError, not NotFound, for these
            raise HTTPException(404, "job not found")

    @web.get("/status/{call_id}")
    async def status(call_id: str):
        check_id(call_id)
        try:
            out = await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except TimeoutError:  # builtin: not finished yet (a job timeout is FunctionTimeoutError, handled below)
            running = await started.contains.aio(call_id)
            return {"id": call_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        except Exception as e:  # the call itself raised (crash, timeout, cancel)
            return _error_status(call_id, e)
        return _status_body(call_id, out)

    @web.post("/cancel/{call_id}")
    async def cancel(call_id: str):
        check_id(call_id)
        try:
            await modal.FunctionCall.from_id(call_id).cancel.aio()
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        return {"id": call_id, "status": "CANCELLED"}

    return web
