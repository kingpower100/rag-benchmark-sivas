FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models/huggingface /models/sentence-transformers /app/logs

COPY pyproject.toml README.md requirements-lock.txt constraints.txt ./
COPY src ./src
COPY configs ./configs
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install torch==2.5.1 \
         --index-url https://download.pytorch.org/whl/cu121 \
    && python3.11 -m pip install --requirement requirements-lock.txt \
    && python3.11 -m pip install --no-deps -e .

CMD ["python3.11", "-m", "src.pipeline1.main", "--config", "configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml"]
