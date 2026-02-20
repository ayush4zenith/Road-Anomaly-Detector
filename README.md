# Road Anomaly Detection on Raspberry Pi 4

Real-time detection of potholes and cracks from dashcam footage using YOLOv5n, optimized for Raspberry Pi 4B with TFLite FP16 quantization.

Built for the **Arm Edge AI** project — runs entirely on CPU at **≥5 FPS** without accelerators.

## Features

- **YOLOv5n** trained on the N-RDD2024 dataset (potholes, manholes, cracks)
- **TFLite FP16** quantized model — 4.94 MB, optimized for Arm Cortex-A72
- **Real-time inference** on Raspberry Pi 4B at ~8-10 FPS (640×352)
- **Docker deployment** with Python 3.10 (Pi OS compatibility)
- **Async logging** — CSV, JSON-Lines, annotated frames, video clips
- **Headless mode** — auto-detects Docker/SSH environments

## Project Structure

```
├── src/
│   ├── main.py                  # Main inference pipeline
│   ├── inference/
│   │   ├── engine.py            # TFLite/ONNX inference engine
│   │   └── kleidi_utils.py      # Arm CPU optimizations
│   └── pipeline/
│       ├── stream.py            # Video capture with frame skipping
│       └── logger.py            # Detection logging and annotation
├── scripts/
│   ├── prepare_data.py          # VOC XML → YOLO format converter
│   ├── train_model.py           # YOLOv5n training script
│   ├── convert_model.py         # Model quantization (FP16/Int8/ONNX)
│   └── setup_pi.sh             # Raspberry Pi setup script
├── models/optimized/
│   └── best-fp16.tflite         # Deployed FP16 TFLite model
├── data/sample_videos/          # Test videos for inference
├── tests/test_all.py            # 72 unit tests
├── Dockerfile                   # Docker config (Python 3.10)
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # Development dependencies
```

## Quick Start (Docker on Raspberry Pi)

```bash
# Build
docker build -t road-anomaly .

# Run on test image
docker run --rm -v ./outputs:/app/outputs road-anomaly \
  --model models/optimized/best-fp16.tflite \
  --source data/sample_videos/test.mp4 --save-frames

# Run with live webcam
docker run --rm --device /dev/video0 -v ./outputs:/app/outputs road-anomaly \
  --model models/optimized/best-fp16.tflite \
  --source 0 --save-frames --save-clips
```

## Performance

| Metric | Value |
|:--|:--|
| Model | YOLOv5n (FP16 TFLite) |
| Model size | 4.94 MB |
| Input resolution | 640 × 352 |
| Inference time | ~75-140ms per frame |
| FPS | ~8-10 FPS on Pi 4B |
| Classes | pothole, manhole, crack |

## Requirements

**Runtime** (Raspberry Pi / Docker):
- Python 3.10
- OpenCV (headless)
- NumPy
- tflite-runtime

**Development** (training/conversion):
- PyTorch + Ultralytics
- TensorFlow (for model conversion)
- ONNX Runtime

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_all.py -v
```
72 tests covering inference engine, logger, NMS, stream, and integration.
