FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY scripts/ scripts/

RUN mkdir -p outputs

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "src/main.py", "--headless"]
CMD ["--save-frames"]
