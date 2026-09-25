"""Pre-build memory snapshots right after a deploy, so real users don't pay for it.

Modal keeps a snapshot per worker type (2-3 per GPU type), drops them on every deploy, and
builds each lazily on the first container that lands there: that request waits +30-40 s.
Calling `warmup` on several containers at once moves that cost here. Coverage isn't
guaranteed (a later request can still land on a new worker type). Each run costs a few
GPU-minutes (the containers then idle for the scaledown window).

Usage: python modal_app/warmup.py [app] [containers]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import modal

app_name = sys.argv[1] if len(sys.argv) > 1 else "demucs"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cls = modal.Cls.from_name(app_name, "Demucs")


def one(i: int) -> str:
    t0 = time.time()
    out = cls().warmup.remote(hold_seconds=20)
    return f"container {i + 1}: {time.time() - t0:.0f}s task={out['task']} device={out['device']}"


with ThreadPoolExecutor(n) as pool:
    for line in pool.map(one, range(n)):
        print(line, flush=True)
