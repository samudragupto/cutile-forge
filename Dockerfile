FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUTILE_ARCH=sm_120

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --no-cache-dir --break-system-packages ".[cuda,serve]"

EXPOSE 8000

ENTRYPOINT ["cutile-forge", "serve"]
CMD ["--model", "/models/mistral_7b_4bit", "--host", "0.0.0.0", "--port", "8000", "--target", "sm_120"]

