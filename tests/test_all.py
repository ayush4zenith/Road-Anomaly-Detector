"""
tests/test_all.py — Combined tests for logger.py and main.py

72 tests covering:
  LOGGER: Detection, FrameLog, colour palette, AnomalyLogger init, CSV,
          JSON-Lines, log_frame, log_event, summary, context manager, close,
          frame saving, clip buffering, edge cases.
  MAIN:   parse_args, NMS, FPSTracker, _FallbackStream, _FallbackEngine,
          main() integration (demo mode), signal handler, CLASS_LABELS.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# ── ensure project root is importable ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.logger import (
    AnomalyLogger, Detection, FrameLog,
    _colour_for, _DEFAULT_COLOUR, _PALETTE,
)
from src.main import (
    CLASS_LABELS, FPSTracker, _FallbackStream, nms, parse_args,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)

def _dets(n=3, conf=0.8):
    labels = ["pothole", "crack", "obstacle"]
    return [
        Detection(label=labels[i % 3], confidence=conf,
                  bbox=(50 + i * 100, 50, 150 + i * 100, 150))
        for i in range(n)
    ]

def _det(label="pothole", conf=0.9, bbox=(10, 10, 100, 100)):
    return Detection(label=label, confidence=conf, bbox=bbox)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGER TESTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Detection dataclass ──
class TestDetection:
    def test_creation(self):
        d = Detection(label="pothole", confidence=0.95, bbox=(10, 20, 100, 200))
        assert d.label == "pothole"
        assert d.confidence == 0.95
        assert d.bbox == (10, 20, 100, 200)
        assert d.frame_index == 0 and d.timestamp == ""

    def test_to_dict(self):
        d = Detection("crack", 0.5, (1, 2, 3, 4), frame_index=7,
                      timestamp="2026-01-01T00:00:00+00:00")
        out = d.to_dict()
        assert isinstance(out, dict) and out["label"] == "crack"
        assert out["confidence"] == 0.5 and out["frame_index"] == 7

    def test_defaults(self):
        d = Detection("x", 0.1, (0, 0, 0, 0))
        assert d.frame_index == 0 and d.timestamp == ""

# ── FrameLog dataclass ──
class TestFrameLog:
    def test_creation(self):
        fl = FrameLog(frame_index=1, timestamp="ts")
        assert fl.detections == [] and fl.inference_ms == 0.0

    def test_to_dict_num_detections(self):
        fl = FrameLog(frame_index=5, timestamp="ts", detections=_dets(2))
        assert fl.to_dict()["num_detections"] == 2

# ── Colour palette ──
class TestColourPalette:
    def test_known(self):
        assert _colour_for("pothole") == _PALETTE["pothole"]
    def test_case_insensitive(self):
        assert _colour_for("CRACK") == _PALETTE["crack"]
    def test_unknown(self):
        assert _colour_for("banana") == _DEFAULT_COLOUR

# ── AnomalyLogger init ──
class TestLoggerInit:
    def test_output_dir(self, tmp_path):
        out = tmp_path / "out"
        AnomalyLogger(output_dir=out).close()
        assert out.is_dir()

    def test_csv_created(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_json=False).close()
        assert len(list(tmp_path.glob("detections_*.csv"))) == 1

    def test_json_created(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_csv=False).close()
        assert len(list(tmp_path.glob("detections_*.jsonl"))) == 1

    def test_no_files_when_disabled(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False).close()
        assert list(tmp_path.glob("detections_*")) == []

    def test_frames_dir(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_annotated_frames=True,
                       save_csv=False, save_json=False).close()
        assert (tmp_path / "frames").is_dir()

    def test_clips_dir(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_clips=True,
                       save_csv=False, save_json=False).close()
        assert (tmp_path / "clips").is_dir()

# ── CSV logging ──
class TestCSVLogging:
    def test_header(self, tmp_path):
        AnomalyLogger(output_dir=tmp_path, save_json=False).close()
        with open(list(tmp_path.glob("*.csv"))[0]) as f:
            hdr = next(csv.reader(f))
        assert hdr == ["frame_index", "timestamp", "label", "confidence",
                       "x_min", "y_min", "x_max", "y_max", "inference_ms", "fps"]

    def test_rows(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_json=False, confidence_threshold=0.1)
        lg.log_frame(_frame(), _dets(2, 0.9), inference_ms=10.5, fps=15.0)
        lg.close()
        with open(list(tmp_path.glob("*.csv"))[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 and rows[0]["label"] == "pothole"
        assert float(rows[0]["inference_ms"]) == 10.5

    def test_below_threshold(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_json=False, confidence_threshold=0.99)
        lg.log_frame(_frame(), _dets(2, 0.5)); lg.close()
        with open(list(tmp_path.glob("*.csv"))[0]) as f:
            assert len(list(csv.DictReader(f))) == 0

# ── JSON-Lines logging ──
class TestJSONLogging:
    def test_entries(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, confidence_threshold=0.1)
        lg.log_frame(_frame(), _dets(3, 0.7), inference_ms=5.0, fps=20.0); lg.close()
        with open(list(tmp_path.glob("*.jsonl"))[0]) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 3 and lines[0]["inference_ms"] == 5.0

    def test_empty(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False)
        lg.log_frame(_frame(), []); lg.close()
        with open(list(tmp_path.glob("*.jsonl"))[0]) as f:
            assert f.read().strip() == ""

# ── log_frame behaviour ──
class TestLogFrame:
    def test_returns_ndarray(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        r = lg.log_frame(_frame(), []); lg.close()
        assert isinstance(r, np.ndarray)

    def test_original_not_mutated(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        f = _frame(); orig = f.copy()
        lg.log_frame(f, _dets(1)); lg.close()
        np.testing.assert_array_equal(f, orig)

    def test_detection_count(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False,
                            confidence_threshold=0.1)
        lg.log_frame(_frame(), _dets(2, 0.8))
        lg.log_frame(_frame(), _dets(3, 0.8))
        s = lg.get_summary(); lg.close()
        assert s["total_detections"] == 5 and s["total_frames_processed"] == 2

    def test_confidence_filter(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False,
                            confidence_threshold=0.6)
        lg.log_frame(_frame(), [
            Detection("pothole", 0.8, (0,0,10,10)),
            Detection("crack",   0.3, (20,20,30,30)),
            Detection("obstacle",0.7, (40,40,50,50)),
        ])
        assert lg.get_summary()["total_detections"] == 2; lg.close()

    def test_fps_accumulation(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        for fps in (10.0, 20.0, 30.0):
            lg.log_frame(_frame(), [], fps=fps)
        assert lg.get_summary()["average_fps"] == pytest.approx(20.0, abs=0.01)
        lg.close()

    def test_class_counts(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False,
                            confidence_threshold=0.1)
        lg.log_frame(_frame(), [
            Detection("pothole", 0.9, (0,0,10,10)),
            Detection("pothole", 0.8, (20,20,30,30)),
            Detection("crack",   0.7, (40,40,50,50)),
        ])
        s = lg.get_summary(); lg.close()
        assert s["detections_per_class"]["pothole"] == 2
        assert s["detections_per_class"]["crack"] == 1

# ── log_event ──
class TestLogEvent:
    def test_event_in_jsonl(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False)
        lg.log_event("test_event", "hello", {"key": "val"}); lg.close()
        with open(list(tmp_path.glob("*.jsonl"))[0]) as f:
            e = json.loads(f.readline())
        assert e["event"] == "test_event" and e["key"] == "val"

    def test_no_crash_without_json(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        lg.log_event("x", "y"); lg.close()

# ── Summary ──
class TestSummary:
    def test_keys(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        s = lg.get_summary(); lg.close()
        for k in ("session_duration_s", "total_frames_processed",
                   "total_detections", "detections_per_class",
                   "average_fps", "output_directory"):
            assert k in s

    def test_save_json(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        lg.log_frame(_frame(), _dets(1, 0.9), fps=12.0)
        p = lg.save_summary_json(); lg.close()
        with open(p) as f:
            assert json.load(f)["total_frames_processed"] == 1

    def test_print(self, tmp_path, capsys):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False)
        lg.log_frame(_frame(), _dets(2), fps=10.0)
        lg.print_summary(); lg.close()
        assert "SESSION SUMMARY" in capsys.readouterr().out

# ── Context manager ──
class TestContextManager:
    def test_with(self, tmp_path):
        with AnomalyLogger(output_dir=tmp_path) as lg:
            lg.log_frame(_frame(), _dets(1))
        assert lg._csv_file is None and lg._json_file is None

# ── Close ──
class TestClose:
    def test_double_close(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path); lg.close(); lg.close()

    def test_handles_none(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path); lg.close()
        assert lg._csv_file is None and lg._json_file is None

# ── Frame saving ──
class TestFrameSaving:
    def test_saved_on_detection(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_annotated_frames=True,
                            save_csv=False, save_json=False, confidence_threshold=0.1)
        lg.log_frame(_frame(), _dets(1, 0.9)); lg.close()
        assert len(list((tmp_path / "frames").glob("*.jpg"))) == 1

    def test_not_saved_without_detection(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_annotated_frames=True,
                            save_csv=False, save_json=False)
        lg.log_frame(_frame(), []); lg.close()
        assert len(list((tmp_path / "frames").glob("*.jpg"))) == 0

# ── Clip buffering ──
class TestClipBuffering:
    def test_clip_created(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_clips=True,
                            save_csv=False, save_json=False,
                            clip_pre_seconds=0.1, clip_post_seconds=0.2,
                            clip_fps=10, confidence_threshold=0.1)
        lg.log_frame(_frame(), _dets(1, 0.9))
        for _ in range(10):
            lg.log_frame(_frame(), [])
        lg.close()
        assert len(list((tmp_path / "clips").glob("*.avi"))) >= 1

# ── Logger edge cases ──
class TestLoggerEdgeCases:
    def test_empty_dets(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path)
        assert isinstance(lg.log_frame(_frame(), []), np.ndarray); lg.close()

    def test_all_below_threshold(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, confidence_threshold=0.99)
        lg.log_frame(_frame(), _dets(5, 0.5))
        assert lg.get_summary()["total_detections"] == 0; lg.close()

    def test_many_frames(self, tmp_path):
        lg = AnomalyLogger(output_dir=tmp_path, save_csv=False, save_json=False,
                            confidence_threshold=0.1)
        f = _frame(320, 240)
        for _ in range(100):
            lg.log_frame(f, _dets(1, 0.9), fps=25.0)
        s = lg.get_summary(); lg.close()
        assert s["total_frames_processed"] == 100 and s["total_detections"] == 100

    def test_draw_static(self):
        f = _frame()
        r = AnomalyLogger._draw_detections(f, _dets(2, 0.75), fps=10.0, inference_ms=5.0)
        assert r.shape == f.shape and not np.array_equal(r, np.zeros_like(r))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN.PY TESTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Detection import ──
class TestDetectionImport:
    def test_consistent(self):
        assert Detection("pothole", 0.5, (0,0,1,1)).label == "pothole"

# ── parse_args ──
class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["main.py"]):
            a = parse_args()
        assert a.source == "0" and a.model == "models/optimized/model.tflite"
        assert a.conf_thresh == 0.25 and a.headless is False

    def test_custom(self):
        with patch("sys.argv", ["main.py", "--source", "v.mp4", "--model", "m.onnx",
                                 "--conf-thresh", "0.5", "--headless", "--save-frames",
                                 "--save-clips", "--no-csv", "--no-json",
                                 "--max-fps", "15", "--warmup", "3"]):
            a = parse_args()
        assert a.source == "v.mp4" and a.conf_thresh == 0.5
        assert a.headless and a.save_frames and a.save_clips
        assert a.no_csv and a.no_json and a.max_fps == 15

# ── NMS ──
class TestNMS:
    def test_empty(self):
        assert nms([]) == []

    def test_single(self):
        assert len(nms([_det()])) == 1

    def test_non_overlapping_kept(self):
        assert len(nms([_det(bbox=(0,0,50,50)), _det(bbox=(200,200,300,300))])) == 2

    def test_overlapping_suppressed(self):
        r = nms([_det(conf=0.9, bbox=(0,0,100,100)),
                 _det(conf=0.7, bbox=(5,5,105,105))], iou_thresh=0.5)
        assert len(r) == 1 and r[0].confidence == 0.9

    def test_diff_classes_kept(self):
        assert len(nms([_det("pothole", bbox=(0,0,100,100)),
                        _det("crack",   bbox=(0,0,100,100))])) == 2

    def test_three_overlap(self):
        r = nms([_det(conf=0.5, bbox=(0,0,100,100)),
                 _det(conf=0.9, bbox=(2,2,102,102)),
                 _det(conf=0.7, bbox=(4,4,104,104))], iou_thresh=0.3)
        assert len(r) == 1 and r[0].confidence == 0.9

    def test_iou_zero(self):
        assert len(nms([_det(conf=0.9, bbox=(0,0,100,100)),
                        _det(conf=0.8, bbox=(50,50,150,150))], iou_thresh=0.0)) == 1

    def test_iou_one(self):
        assert len(nms([_det(conf=0.9, bbox=(0,0,100,100)),
                        _det(conf=0.8, bbox=(5,5,105,105))], iou_thresh=1.0)) == 2

    def test_identical_boxes(self):
        r = nms([_det(conf=c, bbox=(10,10,50,50)) for c in (0.9, 0.8, 0.7)], iou_thresh=0.5)
        assert len(r) == 1 and r[0].confidence == 0.9

    def test_large_batch(self):
        dets = [_det(conf=0.5+i*0.001, bbox=(i,i,i+50,i+50)) for i in range(200)]
        r = nms(dets, iou_thresh=0.5)
        assert 1 <= len(r) <= 200

    def test_mixed_classes(self):
        dets = [_det("pothole", 0.9, (0,0,100,100)),
                _det("pothole", 0.85,(5,5,105,105)),
                _det("crack",   0.95,(0,0,100,100)),
                _det("crack",   0.80,(5,5,105,105)),
                _det("obstacle",0.7, (200,200,300,300))]
        labels = [d.label for d in nms(dets, iou_thresh=0.5)]
        assert labels.count("pothole") == 1
        assert labels.count("crack") == 1
        assert labels.count("obstacle") == 1

# ── FPSTracker ──
class TestFPSTracker:
    def test_initial_zero(self):
        assert FPSTracker().fps == 0.0

    def test_positive_after_tick(self):
        t = FPSTracker(alpha=0.5); time.sleep(0.01)
        assert t.tick() > 0

    def test_ema_converges(self):
        t = FPSTracker(alpha=0.3)
        for _ in range(50):
            time.sleep(0.01); t.tick()
        assert t.fps > 10

    def test_property_matches(self):
        t = FPSTracker(alpha=1.0); time.sleep(0.02)
        v = t.tick(); assert t.fps == v

# ── _FallbackStream ──
class TestFallbackStream:
    def test_invalid_source(self):
        assert _FallbackStream("nonexistent.mp4").open() is False

    def test_read_before_open(self):
        ret, f = _FallbackStream("0").read()
        assert ret is False and f is None

    def test_release_before_open(self):
        _FallbackStream("0").release()

    def test_fps_default(self):
        assert _FallbackStream("0").fps == 30.0

    def test_synthetic_video(self, tmp_path):
        vp = str(tmp_path / "t.avi")
        w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320,240))
        for _ in range(5): w.write(_frame(320,240))
        w.release()
        s = _FallbackStream(vp); assert s.open()
        ret, f = s.read(); s.release()
        assert ret and f is not None and f.shape[:2] == (240, 320)

# ── _FallbackEngine ──
class TestFallbackEngine:
    def test_bad_format(self):
        from src.main import _FallbackEngine
        with pytest.raises(ValueError, match="Unsupported"):
            _FallbackEngine("model.xyz")

# ── main() integration ──
class TestMainIntegration:
    def _make_video(self, tmp_path, name="v.avi", n=3):
        vp = str(tmp_path / name)
        w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320,240))
        for _ in range(n): w.write(_frame(320,240))
        w.release()
        return vp

    def test_demo_mode(self, tmp_path):
        from src.main import main as run_main
        vp = self._make_video(tmp_path)
        out = str(tmp_path / "out")
        with patch("sys.argv", ["main.py", "--source", vp, "--model", "no.tflite",
                                 "--output", out, "--headless", "--warmup", "0"]):
            run_main()
        sf = Path(out) / "session_summary.json"
        assert sf.is_file()
        assert json.load(open(sf))["total_frames_processed"] >= 1

    def test_headless_short(self, tmp_path):
        from src.main import main as run_main
        vp = self._make_video(tmp_path, "s.avi", 5)
        out = str(tmp_path / "out2")
        with patch("sys.argv", ["main.py", "--source", vp, "--model", "no.tflite",
                                 "--output", out, "--headless", "--no-csv", "--warmup", "0"]):
            run_main()
        assert json.load(open(Path(out)/"session_summary.json"))["total_frames_processed"] >= 3

# ── Signal handler ──
class TestSignalHandler:
    def test_sets_false(self):
        import src.main as m
        m._RUNNING = True; m._signal_handler(2, None)
        assert m._RUNNING is False; m._RUNNING = True

# ── CLASS_LABELS ──
class TestClassLabels:
    def test_not_empty(self):
        assert len(CLASS_LABELS) >= 3
    def test_pothole(self):
        assert "pothole" in CLASS_LABELS
    def test_obstacle(self):
        assert "obstacle" in CLASS_LABELS
    def test_all_strings(self):
        assert all(isinstance(l, str) for l in CLASS_LABELS)
