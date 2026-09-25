# RunPod serverless worker for Demucs (music source separation).
# Build from the repo root so the local demucs package is used:
#   docker buildx build --platform linux/amd64 -f runpod/Dockerfile -t <image> .
#
# Everything RunPod-specific lives in runpod/; upstream files are untouched so the fork can
# keep merging facebookresearch/demucs cleanly.
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_HOME=/models/torch \
    PYTHONPATH=/opt/demucs

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the base image's torch/torchaudio; demucs' own pins would downgrade them.
# diffq (for mdx_q / mdx_extra_q) builds a C extension, so a compiler is installed just for this step.
COPY runpod/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install -r /tmp/requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY demucs /opt/demucs/demucs

# Bake every model's weights (~2 GB) into the image so cold starts don't hit dl.fbaipublicfiles.com.
# get_model also loads each one on CPU: a build-time check that the import chain works.
RUN python -c "from demucs.pretrained import get_model; \
[get_model(n) for n in ['htdemucs','htdemucs_ft','htdemucs_6s','hdemucs_mmi','mdx','mdx_q','mdx_extra','mdx_extra_q']]"

WORKDIR /app
COPY runpod/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
