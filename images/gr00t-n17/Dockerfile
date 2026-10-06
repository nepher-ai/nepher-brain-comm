# GR00T N1.7 brain image.
# One image for every tournament that uses this model. No weights are baked in.
# Python, torch, and flash-attn follow Isaac-GR00T's own lock at the pinned commit
# (requires-python == 3.10.*, torch == 2.7.1, flash-attn == 2.7.4.post1, CUDA 12.8).

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NEPHER_BRAIN_CHECK=gr00t_n17_check:check \
    PYTHONPATH=/opt/nepher

ARG GROOT_REF=23ace64f17aa5015259b8609d371eb61a357c776
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev git build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"

RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /opt/isaac-gr00t \
    && git -C /opt/isaac-gr00t checkout "${GROOT_REF}" \
    && python -m pip install --no-cache-dir --no-deps /opt/isaac-gr00t \
    && python -m pip install --no-cache-dir \
        numpy==1.26.4 transformers==4.57.3 diffusers==0.35.1 \
        einops==0.8.1 peft==0.17.1 omegaconf==2.3.0 safetensors

COPY nepher_brain /opt/nepher-brain/nepher_brain
COPY pyproject.toml README.md LICENSE /opt/nepher-brain/
RUN python -m pip install --no-cache-dir --no-deps /opt/nepher-brain \
    && python -m pip install --no-cache-dir "msgpack>=1.0"

COPY images/gr00t-n17/gr00t_n17_check.py images/gr00t-n17/reference_signature.json /opt/nepher/

WORKDIR /submission
ENTRYPOINT ["nepher-brain"]
CMD ["serve", "--replicas", "1"]
