"""TFLite inference engine for road anomaly detection on Raspberry Pi.

This module provides a lightweight TFLite-based inference wrapper optimized
for edge deployment on Arm devices. Supports both quantized (INT8/UINT8)
and float32 models with automatic preprocessing.

Usage:
    engine = TFLiteEngine('models/optimized/detect.tflite', num_threads=4)
    input_tensor = engine.preprocess(frame)  # frame is HxWx3 BGR (OpenCV)
    boxes, classes, scores = engine.predict(input_tensor)
"""

from typing import Tuple

import cv2
import numpy as np

try:
    # On Raspberry Pi it's common to use the tflite_runtime package
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fall back to full TensorFlow if available (desktop/testing)
    import tensorflow as tf
    tflite = tf.lite


class TFLiteEngine:
    """Minimal TFLite-only inference engine for object detection.

    Attributes:
        interpreter: TFLite interpreter instance.
        input_shape: Expected input tensor shape (batch, height, width, channels).
        input_dtype: Expected input data type (np.float32, np.uint8, or np.int8).
    """

    def __init__(self, model_path: str, num_threads: int = 4) -> None:
        """Initialize the TFLite interpreter with the given model.

        Args:
            model_path: Path to the .tflite model file.
            num_threads: Number of CPU threads for inference (default: 4).

        Raises:
            ValueError: If model has fewer than 3 output tensors.
        """
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Validate model has expected outputs for detection
        if len(self.output_details) < 3:
            raise ValueError(
                f"Model has {len(self.output_details)} outputs, expected at least 3 "
                "(boxes, classes, scores)"
            )

        # Cache frequently accessed input properties
        self.input_index: int = input_details[0]['index']
        self.input_shape: Tuple[int, ...] = tuple(input_details[0]['shape'])
        self.input_dtype: np.dtype = input_details[0]['dtype']

        # Extract target dimensions for preprocessing
        self._target_size: Tuple[int, int] = (
            self.input_shape[2],  # width
            self.input_shape[1]   # height
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize an OpenCV BGR frame to model input.

        Args:
            frame: Input image as HxWx3 BGR numpy array (OpenCV format).

        Returns:
            Preprocessed tensor with shape matching model input requirements.
            Dtype is automatically set based on model (float32 or uint8/int8).
        """
        # Resize to model's expected dimensions
        # pylint: disable=no-member
        resized = cv2.resize(frame, self._target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member

        # Expand to batch dimension: (H, W, C) -> (1, H, W, C)
        batched = np.expand_dims(rgb, axis=0)

        # Apply normalization based on model's expected input dtype
        if self.input_dtype == np.float32:
            # Standard normalization: [0, 255] -> [-1, 1]
            return (batched.astype(np.float32) - 127.5) / 127.5

        # Quantized models (uint8/int8) expect raw pixel values
        return batched.astype(self.input_dtype)

    def predict(
        self, input_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and return detection results.

        Args:
            input_tensor: Preprocessed input tensor from preprocess() method.

        Returns:
            Tuple of (boxes, classes, scores):
                - boxes: Detection bounding boxes as (N, 4) array.
                - classes: Class IDs as (N,) int32 array.
                - scores: Confidence scores as (N,) float array.

        Note:
            Output order assumes SSD-style TFLite detection models.
            Adjust indices if your model uses a different output layout.
        """
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        # Extract outputs (typical SSD order: boxes, classes, scores)
        boxes = np.squeeze(
            self.interpreter.get_tensor(self.output_details[0]['index'])
        )
        classes = np.squeeze(
            self.interpreter.get_tensor(self.output_details[1]['index'])
        ).astype(np.int32)
        scores = np.squeeze(
            self.interpreter.get_tensor(self.output_details[2]['index'])
        )

        return boxes, classes, scores


__all__ = ["TFLiteEngine"]
