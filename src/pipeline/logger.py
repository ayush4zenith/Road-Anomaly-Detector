"""
pipeline/logger.py — CSV/JSON Timestamped Anomaly Logging

Provides the AnomalyLogger class that:
  • Writes per-detection rows to a CSV file.
  • Writes per-detection entries to a JSON-Lines (.jsonl) file.
  • Optionally saves annotated frames/clips to disk.
  • Tracks session-level statistics (total frames, detections, FPS).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger (Python logging, not to be confused with AnomalyLogger)
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected anomaly within a frame."""

    label: str                      # e.g. "pothole", "crack", "obstacle"
    confidence: float               # 0.0 – 1.0
    bbox: tuple[int, int, int, int] # (x_min, y_min, x_max, y_max) in pixels
    frame_index: int = 0
    timestamp: str = ""             # ISO-8601 string, filled by logger

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameLog:
    """Aggregated log entry for one processed frame."""

    frame_index: int
    timestamp: str                               # ISO-8601
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0                    # inference time in ms
    fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["num_detections"] = len(self.detections)
        return d


# ---------------------------------------------------------------------------
# Colour palette for bounding-box drawing
# ---------------------------------------------------------------------------
_PALETTE: Dict[str, tuple[int, int, int]] = {
    "pothole":   (0, 0, 255),     # red
    "crack":     (0, 165, 255),   # orange
    "obstacle":  (0, 255, 255),   # yellow
}
_DEFAULT_COLOUR = (0, 255, 0)     # green for unknown classes


def _colour_for(label: str) -> tuple[int, int, int]:
    return _PALETTE.get(label.lower(), _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# AnomalyLogger
# ---------------------------------------------------------------------------

class AnomalyLogger:
    """
    Thread-safe logger that persists anomaly detection results to CSV and
    JSON-Lines files and optionally saves annotated frames/clips.

    Usage
    -----
        logger = AnomalyLogger(output_dir="outputs/run_001")
        ...98l
        logger.log_frame(frame_bgr, detections, inference_ms, fps)
        ...
        logger.close()          # flush everything
        logger.print_summary()  # quick stats to stdout
    """

    # ---- construction / teardown ----

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        save_csv: bool = True,
        save_json: bool = True,
        save_annotated_frames: bool = False,
        save_clips: bool = False,
        clip_pre_seconds: float = 1.0,
        clip_post_seconds: float = 2.0,
        clip_fps: int = 10,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.clips_dir = self.output_dir / "clips"

        self.save_csv = save_csv
        self.save_json = save_json
        self.save_frames = save_annotated_frames
        self.save_clips = save_clips
        self.clip_pre_sec = clip_pre_seconds
        self.clip_post_sec = clip_post_seconds
        self.clip_fps = clip_fps
        self.confidence_threshold = confidence_threshold

        # Session-level bookkeeping
        self._session_start: float = time.time()
        self._total_frames: int = 0
        self._total_detections: int = 0
        self._class_counts: Dict[str, int] = {}
        self._fps_accum: List[float] = []

        # Clip buffering — circular buffer of recent frames
        self._clip_buffer: List[np.ndarray] = []
        self._clip_max_pre_frames: int = int(clip_pre_seconds * clip_fps)
        self._pending_clip_post: int = 0
        self._clip_writer: Optional[cv2.VideoWriter] = None
        self._clip_index: int = 0

        # File handles
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._json_file = None

        self._setup_output_dirs()
        self._open_files()

        log.info("AnomalyLogger initialised → %s", self.output_dir)

    # ---- private setup helpers ----

    def _setup_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.save_clips:
            self.clips_dir.mkdir(parents=True, exist_ok=True)

    def _open_files(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.save_csv:
            csv_path = self.output_dir / f"detections_{ts}.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            fieldnames = [
                "frame_index", "timestamp", "label", "confidence",
                "x_min", "y_min", "x_max", "y_max",
                "inference_ms", "fps",
            ]
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
            log.info("CSV log → %s", csv_path)

        if self.save_json:
            json_path = self.output_dir / f"detections_{ts}.jsonl"
            self._json_file = open(json_path, "w", encoding="utf-8")
            log.info("JSON-Lines log → %s", json_path)

    # ---- public API ----

    def log_frame(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        inference_ms: float = 0.0,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Log all detections for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from the video stream.
        detections : Sequence[Detection]
            List of Detection objects for this frame.
        inference_ms : float
            Time spent on inference for this frame (milliseconds).
        fps : float
            Current running FPS of the pipeline.

        Returns
        -------
        np.ndarray
            Annotated copy of the frame with bounding boxes drawn.
        """
        self._total_frames += 1
        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # Filter by confidence threshold
        valid_dets = [
            d for d in detections if d.confidence >= self.confidence_threshold
        ]

        # Annotate frame
        annotated = self._draw_detections(frame.copy(), valid_dets, fps, inference_ms)

        # Persist each detection
        for det in valid_dets:
            det.frame_index = self._total_frames
            det.timestamp = now_iso
            self._write_csv_row(det, inference_ms, fps)
            self._write_json_entry(det, inference_ms, fps)
            self._total_detections += 1
            self._class_counts[det.label] = self._class_counts.get(det.label, 0) + 1

        # Optionally save the annotated frame image
        if self.save_frames and valid_dets:
            fpath = self.frames_dir / f"frame_{self._total_frames:06d}.jpg"
            cv2.imwrite(str(fpath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Clip buffering
        if self.save_clips:
            self._handle_clip_buffer(annotated, bool(valid_dets))

        self._fps_accum.append(fps)

        return annotated

    def log_event(self, event_type: str, message: str, extra: Optional[Dict] = None) -> None:
        """Log a generic pipeline event (e.g. 'start', 'stop', 'error')."""
        entry = {
            "event": event_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        if extra:
            entry.update(extra)
        if self._json_file:
            self._json_file.write(json.dumps(entry) + "\n")
            self._json_file.flush()
        log.info("[%s] %s", event_type, message)

    def get_summary(self) -> Dict[str, Any]:
        """Return session-level statistics as a dict."""
        elapsed = time.time() - self._session_start
        avg_fps = sum(self._fps_accum) / len(self._fps_accum) if self._fps_accum else 0.0
        return {
            "session_duration_s": round(elapsed, 2),
            "total_frames_processed": self._total_frames,
            "total_detections": self._total_detections,
            "detections_per_class": dict(self._class_counts),
            "average_fps": round(avg_fps, 2),
            "output_directory": str(self.output_dir),
        }

    def print_summary(self) -> None:
        """Pretty-print session statistics to stdout."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("  ANOMALY DETECTION SESSION SUMMARY")
        print("=" * 60)
        print(f"  Duration           : {summary['session_duration_s']:.1f} s")
        print(f"  Frames processed   : {summary['total_frames_processed']}")
        print(f"  Total detections   : {summary['total_detections']}")
        print(f"  Average FPS        : {summary['average_fps']:.1f}")
        if summary["detections_per_class"]:
            print("  Detections by class:")
            for cls, cnt in sorted(summary["detections_per_class"].items()):
                print(f"    • {cls:20s}: {cnt}")
        print(f"  Outputs saved to   : {summary['output_directory']}")
        print("=" * 60 + "\n")

    def save_summary_json(self) -> Path:
        """Write session summary to a JSON file and return its path."""
        summary = self.get_summary()
        path = self.output_dir / "session_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Session summary saved → %s", path)
        return path

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._finish_clip_writer()
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._json_file:
            self._json_file.close()
            self._json_file = None
        log.info("AnomalyLogger closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ---- annotation drawing ----

    @staticmethod
    def _draw_detections(
        frame: np.ndarray,
        detections: Sequence[Detection],
        fps: float = 0.0,
        inference_ms: float = 0.0,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and an FPS overlay on *frame* (mutates in-place)."""
        for det in detections:
            colour = _colour_for(det.label)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            text = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # FPS / inference overlay (top-left)
        overlay = f"FPS: {fps:.1f} | Inf: {inference_ms:.1f}ms"
        cv2.putText(
            frame, overlay, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        return frame

    # ---- CSV / JSON writers ----

    def _write_csv_row(self, det: Detection, inference_ms: float, fps: float) -> None:
        if self._csv_writer is None:
            return
        x1, y1, x2, y2 = det.bbox
        self._csv_writer.writerow({
            "frame_index": det.frame_index,
            "timestamp": det.timestamp,
            "label": det.label,
            "confidence": round(det.confidence, 4),
            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
            "inference_ms": round(inference_ms, 2),
            "fps": round(fps, 2),
        })
        self._csv_file.flush()

    def _write_json_entry(self, det: Detection, inference_ms: float, fps: float) -> None:
        if self._json_file is None:
            return
        entry = det.to_dict()
        entry["inference_ms"] = round(inference_ms, 2)
        entry["fps"] = round(fps, 2)
        self._json_file.write(json.dumps(entry) + "\n")
        self._json_file.flush()

    # ---- clip saving helpers ----

    def _handle_clip_buffer(self, frame: np.ndarray, has_detection: bool) -> None:
        """Maintain a rolling buffer of recent frames and save clips around detections."""
        self._clip_buffer.append(frame.copy())
        if len(self._clip_buffer) > self._clip_max_pre_frames:
            self._clip_buffer.pop(0)

        if has_detection:
            # Start a new clip (or extend the current one)
            if self._clip_writer is None:
                self._start_clip_writer(frame.shape[1], frame.shape[0])
                # Write pre-buffer frames
                for buf_frame in self._clip_buffer:
                    self._clip_writer.write(buf_frame)
            self._pending_clip_post = int(self.clip_post_sec * self.clip_fps)

        elif self._clip_writer is not None:
            self._clip_writer.write(frame)
            self._pending_clip_post -= 1
            if self._pending_clip_post <= 0:
                self._finish_clip_writer()

    def _start_clip_writer(self, width: int, height: int) -> None:
        self._clip_index += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = self.clips_dir / f"clip_{self._clip_index:04d}_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._clip_writer = cv2.VideoWriter(
            str(clip_path), fourcc, self.clip_fps, (width, height),
        )
        log.info("Recording clip → %s", clip_path)

    def _finish_clip_writer(self) -> None:
        if self._clip_writer is not None:
            self._clip_writer.release()
            self._clip_writer = None
            log.info("Clip saved (clip_%04d).", self._clip_index)
