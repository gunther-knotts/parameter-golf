# Dockerfile for Parameter Golf Autoresearch
#
# Builds a self-contained image that can run on RunPod, Lambda, or any
# NVIDIA GPU cloud with the NVIDIA Container Toolkit.
#
# Build:
#   docker build -t parameter-golf-autoresearch -f autoresearch/Dockerfile .
#
# Run (1 GPU, proxy mode):
#   docker run --gpus all -e ANTHROPIC_API_KEY=sk-ant-... \
#     parameter-golf-autoresearch \
#     python3 autoresearch/autoresearch_golf.py --max-experiments 50
#
# Run (8 GPU, full mode):
#   docker run --gpus all --shm-size=1g -e ANTHROPIC_API_KEY=sk-ant-... \
#     parameter-golf-autoresearch \
#     python3 autoresearch/autoresearch_golf.py --max-experiments 50 --full-ngpu 8
#
# Mount results volume to persist across runs:
#   docker run --gpus all \
#     -v $(pwd)/autoresearch_results:/workspace/parameter-golf/autoresearch \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     parameter-golf-autoresearch \
#     python3 autoresearch/autoresearch_golf.py --resume

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    numpy \
    torch==2.10 \
    sentencepiece \
    huggingface-hub \
    datasets \
    tqdm \
    anthropic \
    setuptools \
    typing-extensions==4.15.0

WORKDIR /workspace/parameter-golf

# Copy repo
COPY . .

# Make scripts executable
RUN chmod +x autoresearch/run_experiment.sh autoresearch/setup_runpod.sh

# Download dataset at build time (validation + 10 training shards)
# For full training, override with --train-shards 80 or mount a volume
RUN python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Default: run the harness
ENTRYPOINT ["python3", "autoresearch/autoresearch_golf.py"]
CMD ["--agent", "claude", "--max-experiments", "100"]
