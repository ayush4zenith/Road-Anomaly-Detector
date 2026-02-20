#!/usr/bin/env python3
"""
convert_model.py - Model Quantization and Optimization for Edge Deployment

Converts object detection models to edge-optimized formats:
- TensorFlow Lite with Int8/FP16 quantization
- ONNX optimization for ONNX Runtime
- YOLOv5 export to TFLite

Int8 quantization is key to achieving ≥5 FPS on Raspberry Pi CPU.

Usage: python scripts/convert_model.py --input model.onnx --output model.tflite --format tflite --quantize int8
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CalibrationDataLoader:
    """Loads calibration images for Int8 quantization."""
    
    def __init__(self, data_dir: Path, input_shape: Tuple[int, int] = (320, 320), num_samples: int = 100):
        self.data_dir = Path(data_dir)
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def _find_images(self) -> List[Path]:
        images = []
        if self.data_dir.exists():
            for ext in self.image_extensions:
                images.extend(self.data_dir.glob(f'*{ext}'))
                images.extend(self.data_dir.glob(f'*{ext.upper()}'))
        return images[:self.num_samples]
    
    def _load_and_preprocess(self, image_path: Path) -> np.ndarray:
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.input_shape[1], self.input_shape[0]))
            img_array = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            return None
    
    def get_representative_dataset(self) -> Callable[[], Generator]:
        """Returns generator function for TFLite representative dataset."""
        images = self._find_images()
        
        if not images:
            logger.warning("No calibration images found. Using random data.")
            def random_data_gen():
                for _ in range(self.num_samples):
                    yield [np.random.rand(1, *self.input_shape, 3).astype(np.float32)]
            return random_data_gen
        
        def data_gen():
            for img_path in images:
                data = self._load_and_preprocess(img_path)
                if data is not None:
                    yield [data]
        return data_gen
    
    def load_samples(self, num: int = 10) -> List[np.ndarray]:
        images = self._find_images()[:num]
        samples = []
        for img_path in images:
            data = self._load_and_preprocess(img_path)
            if data is not None:
                samples.append(data)
        if not samples:
            return [np.random.rand(1, *self.input_shape, 3).astype(np.float32)]
        return samples


class TFLiteConverter:
    """Converts models to TensorFlow Lite format with Int8/FP16/FP32 quantization."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
    
    def _onnx_to_tf(self) -> str:
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            logger.info(f"Loading ONNX model: {self.input_path}")
            onnx_model = onnx.load(str(self.input_path))
            tf_rep = prepare(onnx_model)
            
            tf_path = self.input_path.parent / f"{self.input_path.stem}_tf"
            tf_rep.export_graph(str(tf_path))
            logger.info(f"Converted to TensorFlow SavedModel: {tf_path}")
            return str(tf_path)
        except ImportError:
            logger.error("onnx-tf not installed. Run: pip install onnx-tf")
            raise
    
    def _get_converter(self):
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            raise
        
        if self.input_path.suffix == '.onnx':
            tf_path = self._onnx_to_tf()
            return tf.lite.TFLiteConverter.from_saved_model(tf_path)
        elif self.input_path.suffix == '.h5':
            model = tf.keras.models.load_model(str(self.input_path))
            return tf.lite.TFLiteConverter.from_keras_model(model)
        elif self.input_path.is_dir():
            return tf.lite.TFLiteConverter.from_saved_model(str(self.input_path))
        else:
            raise ValueError(f"Unsupported input format: {self.input_path.suffix}")
    
    def convert_fp32(self) -> Path:
        logger.info("Converting to FP32 TFLite")
        converter = self._get_converter()
        tflite_model = converter.convert()
        self.output_path.write_bytes(tflite_model)
        logger.info(f"Saved: {self.output_path} ({len(tflite_model) / 1e6:.2f} MB)")
        return self.output_path
    
    def convert_fp16(self) -> Path:
        import tensorflow as tf
        logger.info("Converting to FP16 TFLite")
        converter = self._get_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        self.output_path.write_bytes(tflite_model)
        logger.info(f"Saved: {self.output_path} ({len(tflite_model) / 1e6:.2f} MB)")
        return self.output_path
    
    def convert_int8(self, calibration_data: Optional[CalibrationDataLoader] = None) -> Path:
        """Int8 quantization - 4x smaller, fastest inference."""
        import tensorflow as tf
        logger.info("Converting to Int8 TFLite")
        
        converter = self._get_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if calibration_data:
            converter.representative_dataset = calibration_data.get_representative_dataset()
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            logger.info("Using calibration data for full integer quantization")
        else:
            logger.info("No calibration data - using dynamic range quantization")
        
        tflite_model = converter.convert()
        self.output_path.write_bytes(tflite_model)
        logger.info(f"Saved: {self.output_path} ({len(tflite_model) / 1e6:.2f} MB)")
        return self.output_path
    
    def convert(self, quantize: str = 'int8', calibration_data: Optional[CalibrationDataLoader] = None) -> Path:
        if quantize == 'int8':
            return self.convert_int8(calibration_data)
        elif quantize == 'fp16':
            return self.convert_fp16()
        else:
            return self.convert_fp32()


class ONNXOptimizer:
    """Optimizes ONNX models for inference with graph optimization and Int8 quantization."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
    
    def optimize_graph(self) -> Path:
        try:
            import onnx
            from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            raise
        
        logger.info("Applying ONNX graph optimizations")
        model = onnx.load(str(self.input_path))
        onnx.checker.check_model(model)
        
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(self.output_path)
        
        _ = InferenceSession(str(self.input_path), sess_options, providers=['CPUExecutionProvider'])
        logger.info(f"Saved optimized ONNX: {self.output_path}")
        return self.output_path
    
    def quantize_int8(self, calibration_data: Optional[CalibrationDataLoader] = None) -> Path:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            raise
        
        logger.info("Quantizing ONNX model to Int8")
        quantize_dynamic(model_input=str(self.input_path), model_output=str(self.output_path), weight_type=QuantType.QInt8)
        
        output_size = self.output_path.stat().st_size / 1e6
        logger.info(f"Saved Int8 ONNX: {self.output_path} ({output_size:.2f} MB)")
        return self.output_path
    
    def simplify(self) -> Path:
        try:
            import onnx
            from onnxsim import simplify
        except ImportError:
            logger.error("onnx-simplifier not installed. Run: pip install onnx-simplifier")
            raise
        
        logger.info("Simplifying ONNX model")
        model = onnx.load(str(self.input_path))
        model_simp, check = simplify(model)
        
        if not check:
            logger.warning("Simplification validation failed, saving anyway")
        onnx.save(model_simp, str(self.output_path))
        logger.info(f"Saved simplified ONNX: {self.output_path}")
        return self.output_path
    
    def optimize(self, quantize: str = 'int8') -> Path:
        optimized = self.optimize_graph()
        if quantize == 'int8':
            self.input_path = optimized
            return self.quantize_int8()
        return optimized


class YOLOv5Exporter:
    """Exports YOLOv5 models to edge-optimized formats (TFLite/ONNX)."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
    
    def export_to_onnx(self, imgsz: int = 320) -> Path:
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed. Run: pip install torch")
            raise
        
        logger.info(f"Exporting YOLOv5 to ONNX (imgsz={imgsz})")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.input_path))
        model.eval()
        
        dummy_input = torch.zeros(1, 3, imgsz, imgsz)
        onnx_path = self.output_path.with_suffix('.onnx')
        
        torch.onnx.export(
            model.model, dummy_input, str(onnx_path),
            opset_version=12, input_names=['images'], output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        logger.info(f"Saved ONNX model: {onnx_path}")
        return onnx_path
    
    def export_to_tflite(self, imgsz: int = 320, quantize: str = 'int8', calibration_dir: Optional[Path] = None) -> Path:
        onnx_path = self.export_to_onnx(imgsz)
        
        calibration_data = None
        if calibration_dir and quantize == 'int8':
            calibration_data = CalibrationDataLoader(calibration_dir, input_shape=(imgsz, imgsz))
        
        converter = TFLiteConverter(onnx_path, self.output_path)
        return converter.convert(quantize, calibration_data)


def get_model_info(model_path: Path) -> dict:
    """Get information about a model file."""
    model_path = Path(model_path)
    info = {
        'path': str(model_path),
        'format': model_path.suffix,
        'size_mb': model_path.stat().st_size / 1e6 if model_path.exists() else 0
    }
    
    if model_path.suffix == '.tflite':
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            info['input_shape'] = input_details[0]['shape'].tolist()
            info['input_dtype'] = str(input_details[0]['dtype'])
            info['output_shape'] = output_details[0]['shape'].tolist()
            info['output_dtype'] = str(output_details[0]['dtype'])
        except Exception as e:
            logger.warning(f"Could not read TFLite info: {e}")
    
    elif model_path.suffix == '.onnx':
        try:
            import onnx
            model = onnx.load(str(model_path))
            info['opset_version'] = model.opset_import[0].version
            info['input_name'] = model.graph.input[0].name
            info['output_name'] = model.graph.output[0].name
        except Exception as e:
            logger.warning(f"Could not read ONNX info: {e}")
    
    return info


def main():
    parser = argparse.ArgumentParser(
        description='Convert and quantize models for edge deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_model.py --input model.onnx --output model.tflite --format tflite --quantize int8
  python convert_model.py --input model.onnx --output model.tflite --quantize int8 --calibration-data ./images/
  python convert_model.py --input yolov5s.pt --output yolov5s.tflite --format tflite --quantize int8
        """
    )
    
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input model path')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output model path')
    parser.add_argument('--format', '-f', choices=['tflite', 'onnx'], default='tflite', help='Output format')
    parser.add_argument('--quantize', '-q', choices=['int8', 'fp16', 'none'], default='int8', help='Quantization type')
    parser.add_argument('--calibration-data', '-c', type=Path, default=None, help='Calibration images for Int8')
    parser.add_argument('--imgsz', type=int, default=320, help='Input image size for YOLOv5')
    parser.add_argument('--info', action='store_true', help='Print model info and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.info:
        if args.input.exists():
            info = get_model_info(args.input)
            print("\n📊 Model Info:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ Model not found: {args.input}")
        return 0
    
    if not args.input.exists():
        logger.error(f"Input model not found: {args.input}")
        return 1
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    calibration_data = None
    if args.calibration_data and args.quantize == 'int8':
        if args.calibration_data.exists():
            calibration_data = CalibrationDataLoader(args.calibration_data, input_shape=(args.imgsz, args.imgsz))
            logger.info(f"Loaded calibration data from: {args.calibration_data}")
        else:
            logger.warning(f"Calibration directory not found: {args.calibration_data}")
    
    try:
        input_suffix = args.input.suffix.lower()
        
        if input_suffix == '.pt':
            exporter = YOLOv5Exporter(args.input, args.output)
            if args.format == 'tflite':
                output = exporter.export_to_tflite(imgsz=args.imgsz, quantize=args.quantize, calibration_dir=args.calibration_data)
            else:
                output = exporter.export_to_onnx(imgsz=args.imgsz)
        elif args.format == 'tflite':
            converter = TFLiteConverter(args.input, args.output)
            output = converter.convert(args.quantize, calibration_data)
        elif args.format == 'onnx':
            optimizer = ONNXOptimizer(args.input, args.output)
            output = optimizer.optimize(args.quantize)
        else:
            logger.error(f"Unsupported conversion: {input_suffix} -> {args.format}")
            return 1
        
        print("\n" + "=" * 60)
        print("✅ Conversion Complete!")
        print("=" * 60)
        
        info = get_model_info(output)
        print(f"\n📁 Output: {output}")
        print(f"📊 Size: {info['size_mb']:.2f} MB")
        print(f"🔧 Quantization: {args.quantize}")
        
        if 'input_shape' in info:
            print(f"📥 Input: {info['input_shape']} ({info['input_dtype']})")
        if 'output_shape' in info:
            print(f"📤 Output: {info['output_shape']} ({info['output_dtype']})")
        print()
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
