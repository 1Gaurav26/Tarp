# CUDA 12.1 + Python 3.10 base (matches PyTorch 2.x requirements)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for wntr / scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/

# Make data/results/checkpoints directories available for volume mounts
RUN mkdir -p /app/results /app/checkpoints

ENV PYTHONPATH=/app

CMD ["bash"]
