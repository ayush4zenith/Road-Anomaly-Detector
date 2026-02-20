#!/usr/bin/env python3
"""
main.py — Application Entry Point for Real-Time Road Anomaly Detection

Ties together the video stream, inference engine, and anomaly logger into a
single pipeline loop optimised for Raspberry Pi 4/5 (Arm Cortex-A72/A76).

Usage
-----
    # Run on a live camera (default /dev/video0)
    python main.py

    # Run on a sample video file
    python main.py --source data/sample_videos/day.mp4

    # Specify a custom model and output directory
    python main.py --model models/optimized/yolov5s_int8.tflite \
                   --source 0 --output outputs/run_001

    # Headless mode (no display window) for SSH / CI
    python main.py --source data/sample_videos/rain.mp4 --headless

    # Show all options
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Add project root to sys.path so that `src.*` imports work when executed
# as `python src/main.py` from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.logger import AnomalyLogger, Detection  # noqa: E402

# Optional imports — these modules may not be implemented yet.
# When they are, main.py will use them automatically.
try:
    from src.inference.engine import InferenceEngine     # noqa: E402
    _HAS_ENGINE = True
except (ImportError, Exception):
    _HAS_ENGINE = False

try:
    from src.pipeline.stream import VideoStream          # noqa: E402
    _HAS_STREAM = True
except (ImportError, Exception):
    _HAS_STREAM = False

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_RUNNING = True


def _signal_handler(signum, frame):
    global _RUNNING
    log.info("Received signal %s — shutting down …", signum)
    _RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Fallback lightweight inference (when engine.py is not yet implemented)
# ---------------------------------------------------------------------------

# Class labels expected by the road-damage detection models
CLASS_LABELS = [
    "pothole", "longitudinal_crack", "transverse_crack",
    "alligator_crack", "obstacle",
]


class _FallbackEngine:
    """
    Minimal stand-in inference engine used when `src.inference.engine` is
    not yet available.  Loads a TFLite or ONNX model directly so that the
    full pipeline can be demonstrated end-to-end.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self._interpreter = None
        self._onnx_session = None
        self._input_details = None
        self._output_details = None
        self._input_size: Tuple[int, int] = (320, 320)  # H, W default

        self._load_model()

    # ---- model loading ----

    def _load_model(self) -> None:
        ext = Path(self.model_path).suffix.lower()

        if ext == ".tflite":
            self._load_tflite()
        elif ext == ".onnx":
            self._load_onnx()
        else:
            raise ValueError(
                f"Unsupported model format '{ext}'. Use .tflite or .onnx"
            )

    def _load_tflite(self) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            # Fall back to the full TF package
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self._interpreter = Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        _, h, w, _ = self._input_details[0]["shape"]
        self._input_size = (int(h), int(w))
        log.info(
            "TFLite model loaded (%s) — input %s",
            self.model_path, self._input_size,
        )

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(
            self.model_path, providers=providers,
        )
        inp = self._onnx_session.get_inputs()[0]
        # Shape is typically [1, 3, H, W] or [1, H, W, 3]
        shape = inp.shape
        if shape[1] == 3:
            _, _, h, w = shape
        else:
            _, h, w, _ = shape
        self._input_size = (int(h), int(w))
        log.info(
            "ONNX model loaded (%s) — input %s",
            self.model_path, self._input_size,
        )

    # ---- preprocessing ----

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalise *frame* (BGR uint8) to model input tensor."""
        h, w = self._input_size
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._interpreter:
            dtype = self._input_details[0]["dtype"]
            if dtype == np.uint8:
                return np.expand_dims(img, axis=0).astype(np.uint8)
            img = img.astype(np.float32) / 255.0
            return np.expand_dims(img, axis=0)

        # ONNX — typically float32, NCHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)

    # ---- inference ----

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and return a list of Detection objects."""
        orig_h, orig_w = frame.shape[:2]
        tensor = self.preprocess(frame)

        if self._interpreter:
            return self._infer_tflite(tensor, orig_w, orig_h)
        elif self._onnx_session:
            return self._infer_onnx(tensor, orig_w, orig_h)
        return []

    def _infer_tflite(
        self, tensor: np.ndarray, orig_w: int, orig_h: int,
    ) -> List[Detection]:
        interp = self._interpreter
        interp.set_tensor(self._input_details[0]["index"], tensor)
        interp.invoke()

        # Standard SSD output: boxes, classes, scores, num_detections
        outputs = [interp.get_tensor(d["index"]) for d in self._output_details]
        return self._parse_ssd_outputs(outputs, orig_w, orig_h)

    def _infer_onnx(
        self, tensor: np.ndarray, orig_w: int, orig_h: int,
    ) -> List[Detection]:
        sess = self._onnx_session
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: tensor})

        # Try SSD-style first, fall back to YOLO-style
        if len(outputs) >= 4:
            return self._parse_ssd_outputs(outputs, orig_w, orig_h)
        return self._parse_yolo_outputs(outputs, orig_w, orig_h)

    # ---- output parsing helpers ----

    def _parse_ssd_outputs(
        self, outputs: list, orig_w: int, orig_h: int,
    ) -> List[Detection]:
        """Parse standard SSD MobileNet output format."""
        detections: List[Detection] = []

        try:
            boxes = np.squeeze(outputs[0])     # [N, 4] normalised
            classes = np.squeeze(outputs[1]).astype(int)
            scores = np.squeeze(outputs[2])
            num = int(np.squeeze(outputs[3]))
        except (IndexError, ValueError):
            return detections

        for i in range(min(num, len(scores))):
            if scores[i] < self.conf_thresh:
                continue
            y1, x1, y2, x2 = boxes[i]
            bbox = (
                int(x1 * orig_w), int(y1 * orig_h),
                int(x2 * orig_w), int(y2 * orig_h),
            )
            cls_id = int(classes[i])
            label = (
                CLASS_LABELS[cls_id]
                if cls_id < len(CLASS_LABELS)
                else f"class_{cls_id}"
            )
            detections.append(Detection(
                label=label,
                confidence=float(scores[i]),
                bbox=bbox,
            ))
        return detections

    def _parse_yolo_outputs(
        self, outputs: list, orig_w: int, orig_h: int,
    ) -> List[Detection]:
        """Parse YOLO-style [1, N, 5+C] output tensor."""
        detections: List[Detection] = []
        preds = np.squeeze(outputs[0])  # [N, 5+C]

        if preds.ndim != 2 or preds.shape[1] < 6:
            return detections

        for row in preds:
            obj_conf = row[4]
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            score = float(obj_conf * cls_scores[cls_id])
            if score < self.conf_thresh:
                continue

            cx, cy, w, h = row[:4]
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)

            label = (
                CLASS_LABELS[cls_id]
                if cls_id < len(CLASS_LABELS)
                else f"class_{cls_id}"
            )
            detections.append(Detection(
                label=label, confidence=score, bbox=(x1, y1, x2, y2),
            ))
        return detections


# ---------------------------------------------------------------------------
# Fallback video stream (when stream.py is not yet implemented)
# ---------------------------------------------------------------------------

class _FallbackStream:
    """Thin wrapper around cv2.VideoCapture."""

    def __init__(self, source):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            log.error("Cannot open video source: %s", self._source)
            return False
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        log.info("Video source opened: %s (%dx%d @ %.1f fps)", self._source, w, h, fps)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time Road Anomaly Detection on Raspberry Pi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (0, 1 …) or path to video file.",
    )
    p.add_argument(
        "--model", type=str, default="models/optimized/model.tflite",
        help="Path to the optimised .tflite or .onnx model file.",
    )
    p.add_argument(
        "--output", type=str, default="outputs",
        help="Directory for logs, annotated frames, and clips.",
    )

    # Detection
    p.add_argument(
        "--conf-thresh", type=float, default=0.25,
        help="Minimum confidence score to accept a detection.",
    )
    p.add_argument(
        "--iou-thresh", type=float, default=0.45,
        help="IoU threshold for Non-Max Suppression (if applicable).",
    )

    # Output toggles
    p.add_argument("--headless", action="store_true",
                   help="Run without opening a display window.")
    p.add_argument("--save-frames", action="store_true",
                   help="Save annotated frames that contain detections.")
    p.add_argument("--save-clips", action="store_true",
                   help="Save short video clips around detection events.")
    p.add_argument("--no-csv", action="store_true",
                   help="Disable CSV logging.")
    p.add_argument("--no-json", action="store_true",
                   help="Disable JSON-Lines logging.")

    # Performance
    p.add_argument(
        "--max-fps", type=int, default=0,
        help="Cap pipeline FPS (0 = unlimited). Useful for reproducible demos.",
    )
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warm-up inference passes before the main loop.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (simple per-class)
# ---------------------------------------------------------------------------

def nms(detections: List[Detection], iou_thresh: float = 0.45) -> List[Detection]:
    """Apply per-class greedy NMS to a list of Detection objects."""
    if not detections:
        return []

    by_class: dict[str, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.label, []).append(d)

    kept: List[Detection] = []
    for dets in by_class.values():
        dets.sort(key=lambda d: d.confidence, reverse=True)
        boxes = np.array([d.bbox for d in dets], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = list(range(len(dets)))
        keep: List[int] = []

        while order:
            i = order.pop(0)
            keep.append(i)
            remaining = []
            for j in order:
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                iou = inter / (areas[i] + areas[j] - inter + 1e-6)
                if iou < iou_thresh:
                    remaining.append(j)
            order = remaining

        kept.extend(dets[k] for k in keep)
    return kept


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    """Exponential-moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._avg: float = 0.0
        self._last: float = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        instant = 1.0 / dt if dt > 0 else 0.0
        self._avg = self._alpha * instant + (1 - self._alpha) * self._avg
        return self._avg

    @property
    def fps(self) -> float:
        return self._avg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Initialise inference engine ----
    model_path = str(Path(args.model))
    if not Path(model_path).is_file():
        log.warning(
            "Model file not found: %s — running in DEMO mode "
            "(no inference, pipeline & logger will still operate).",
            model_path,
        )
        engine = None
    else:
        if _HAS_ENGINE:
            log.info("Using project InferenceEngine from src.inference.engine")
            engine = InferenceEngine(model_path, confidence_threshold=args.conf_thresh)
        else:
            log.info("Using built-in fallback inference engine")
            engine = _FallbackEngine(model_path, confidence_threshold=args.conf_thresh)

    # ---- Initialise video stream ----
    if _HAS_STREAM:
        log.info("Using project VideoStream from src.pipeline.stream")
        stream = VideoStream(args.source)
    else:
        log.info("Using built-in fallback video stream")
        stream = _FallbackStream(args.source)

    if not stream.open():
        log.error("Failed to open video source '%s'. Exiting.", args.source)
        sys.exit(1)

    # ---- Initialise logger ----
    anomaly_logger = AnomalyLogger(
        output_dir=args.output,
        save_csv=not args.no_csv,
        save_json=not args.no_json,
        save_annotated_frames=args.save_frames,
        save_clips=args.save_clips,
        confidence_threshold=args.conf_thresh,
    )
    anomaly_logger.log_event("session_start", "Pipeline initialised", {
        "model": model_path,
        "source": args.source,
        "confidence_threshold": args.conf_thresh,
    })

    # ---- Warm-up passes (primes caches, JIT, etc.) ----
    if engine is not None and args.warmup > 0:
        log.info("Running %d warm-up inference passes …", args.warmup)
        ret, warmup_frame = stream.read()
        if ret and warmup_frame is not None:
            for _ in range(args.warmup):
                engine.infer(warmup_frame)
        log.info("Warm-up complete.")

    # ---- Main loop ----
    fps_tracker = FPSTracker()
    frame_delay = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    global _RUNNING

    log.info("Starting detection loop … (press 'q' or Ctrl+C to stop)")

    try:
        while _RUNNING:
            loop_start = time.perf_counter()

            ret, frame = stream.read()
            if not ret or frame is None:
                log.info("End of video stream.")
                break

            # -- Inference --
            detections: List[Detection] = []
            inference_ms: float = 0.0

            if engine is not None:
                t0 = time.perf_counter()
                detections = engine.infer(frame)
                inference_ms = (time.perf_counter() - t0) * 1000.0

                # Apply NMS
                detections = nms(detections, iou_thresh=args.iou_thresh)

            # -- FPS tracking --
            current_fps = fps_tracker.tick()

            # -- Log & annotate --
            annotated = anomaly_logger.log_frame(
                frame, detections, inference_ms=inference_ms, fps=current_fps,
            )

            # -- Display --
            if not args.headless:
                cv2.imshow("Road Anomaly Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("User pressed 'q' — stopping.")
                    break

            # -- FPS cap --
            if frame_delay > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping …")

    finally:
        # ---- Cleanup ----
        log.info("Cleaning up …")
        stream.release()
        if not args.headless:
            cv2.destroyAllWindows()

        anomaly_logger.log_event("session_end", "Pipeline stopped")
        anomaly_logger.save_summary_json()
        anomaly_logger.print_summary()
        anomaly_logger.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
