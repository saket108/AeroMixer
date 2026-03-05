FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_lock.txt /app/requirements_lock.txt

RUN python -m pip install --upgrade pip setuptools==69.5.1 wheel && \
    sed '/CLIP.git/d' requirements_lock.txt > reqs.txt && \
    pip install -r reqs.txt && \
    git clone --depth 1 https://github.com/openai/CLIP.git /tmp/CLIP && \
    pip install /tmp/CLIP && \
    rm -rf /tmp/CLIP reqs.txt

COPY . /app

ENTRYPOINT ["python", "scripts/inference_pipeline.py"]
