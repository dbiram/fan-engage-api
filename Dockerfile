FROM python:3.12-slim

RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# === PyTorch NIGHTLY with CUDA 12.8 (includes SM 12.0 / Blackwell) ===
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
RUN pip install --no-cache-dir --pre --index-url $TORCH_INDEX_URL \
    torch

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
