"""
kleidi_utils.py - Arm/KleidiAI Optimizations for Edge Inference

Provides CPU-specific optimizations for Raspberry Pi 4/5 (Cortex-A72/A76):
- Platform detection (Arm architecture, NEON support)
- Thread affinity and CPU core management
- Thermal monitoring to prevent throttling (>80°C)
- Memory alignment for NEON SIMD operations
- Quantization helpers for Int8/FP16 models
"""

import os
import platform
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class ArmPlatformInfo:
    """Detects Arm platform characteristics for optimization decisions."""
    
    def __init__(self):
        self._arch = platform.machine().lower()
        self._is_arm = self._arch in ('aarch64', 'arm64', 'armv7l', 'armv8l')
        self._pi_model = self._detect_pi_model()
        self._cpu_count = os.cpu_count() or 4
        self._has_neon = self._detect_neon_support()
    
    def _detect_pi_model(self) -> Optional[str]:
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            try:
                model = model_path.read_text().strip('\x00').strip()
                if "Raspberry Pi 5" in model:
                    return "Pi5"
                elif "Raspberry Pi 4" in model:
                    return "Pi4"
                elif "Raspberry Pi" in model:
                    return "Pi"
                return model
            except (IOError, PermissionError):
                pass
        return None
    
    def _detect_neon_support(self) -> bool:
        if not self._is_arm:
            return False
        
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            try:
                cpuinfo = cpuinfo_path.read_text()
                return 'neon' in cpuinfo.lower() or 'asimd' in cpuinfo.lower()
            except (IOError, PermissionError):
                pass
        
        return self._arch in ('aarch64', 'arm64')
    
    @property
    def is_arm(self) -> bool:
        return self._is_arm
    
    @property
    def is_raspberry_pi(self) -> bool:
        return self._pi_model is not None
    
    @property
    def pi_model(self) -> Optional[str]:
        return self._pi_model
    
    @property
    def has_neon(self) -> bool:
        return self._has_neon
    
    @property
    def cpu_count(self) -> int:
        return self._cpu_count
    
    @property
    def architecture(self) -> str:
        return self._arch
    
    def get_summary(self) -> dict:
        return {
            "architecture": self._arch,
            "is_arm": self._is_arm,
            "is_raspberry_pi": self.is_raspberry_pi,
            "pi_model": self._pi_model,
            "cpu_count": self._cpu_count,
            "has_neon": self._has_neon,
        }
    
    def __repr__(self) -> str:
        if self.is_raspberry_pi:
            return f"ArmPlatformInfo({self._pi_model}, cores={self._cpu_count}, NEON={self._has_neon})"
        return f"ArmPlatformInfo(arch={self._arch}, cores={self._cpu_count})"


class ThermalMonitor:
    """
    Monitors CPU temperature to prevent thermal throttling.
    Critical: >80°C causes throttling (5 FPS -> ~1 FPS)
    """
    
    TEMP_NORMAL = 70.0
    TEMP_WARNING = 80.0
    TEMP_CRITICAL = 85.0
    
    def __init__(self):
        self._thermal_path = self._find_thermal_path()
        self._last_temp = 0.0
        self._throttle_count = 0
    
    def _find_thermal_path(self) -> Optional[Path]:
        primary = Path("/sys/class/thermal/thermal_zone0/temp")
        if primary.exists():
            return primary
        
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            for zone in thermal_base.glob("thermal_zone*/temp"):
                return zone
        return None
    
    def get_temperature(self) -> float:
        """Read current CPU temperature in Celsius. Returns -1.0 if unavailable."""
        if self._thermal_path is None:
            return -1.0
        
        try:
            temp_raw = self._thermal_path.read_text().strip()
            self._last_temp = float(temp_raw) / 1000.0
            return self._last_temp
        except (IOError, ValueError, PermissionError):
            return -1.0
    
    def get_status(self) -> Tuple[str, float]:
        """Returns (status_string, temperature). Status: 'normal', 'warning', 'critical', 'unknown'"""
        temp = self.get_temperature()
        
        if temp < 0:
            return ('unknown', temp)
        elif temp < self.TEMP_NORMAL:
            return ('normal', temp)
        elif temp < self.TEMP_WARNING:
            return ('warning', temp)
        else:
            self._throttle_count += 1
            return ('critical', temp)
    
    def is_throttling(self) -> bool:
        temp = self.get_temperature()
        return temp >= self.TEMP_WARNING
    
    def should_skip_frame(self) -> bool:
        """Returns True if temperature is critical - use for adaptive frame rate."""
        status, _ = self.get_status()
        return status == 'critical'
    
    @property
    def throttle_count(self) -> int:
        return self._throttle_count
    
    def get_vcgencmd_temp(self) -> Optional[float]:
        """Get temperature via vcgencmd (Raspberry Pi specific)."""
        try:
            import subprocess
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp_val = temp_str.replace("temp=", "").replace("'C", "")
                return float(temp_val)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None


class CpuOptimizer:
    """CPU-specific optimizations for inference performance on Cortex-A72/A76."""
    
    def __init__(self, platform_info: Optional[ArmPlatformInfo] = None):
        self._platform = platform_info or ArmPlatformInfo()
    
    def get_optimal_num_threads(self) -> int:
        """Returns recommended thread count for TFLite/ONNX (4 for Pi 4/5)."""
        cpu_count = self._platform.cpu_count
        
        if self._platform.is_raspberry_pi:
            return min(cpu_count, 4)
        
        return max(1, cpu_count - 1)
    
    def set_process_priority(self, priority: str = 'high') -> bool:
        """Set process priority. Args: 'high', 'normal', 'low'. Returns True if successful."""
        nice_values = {'high': -10, 'normal': 0, 'low': 10}
        nice_val = nice_values.get(priority, 0)
        
        try:
            os.nice(nice_val)
            return True
        except (PermissionError, OSError):
            return False
    
    def get_cpu_freq_info(self) -> dict:
        """Get CPU frequency info in MHz."""
        freq_info = {'current': None, 'min': None, 'max': None}
        freq_base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
        
        if not freq_base.exists():
            return freq_info
        
        try:
            for key, filename in [('current', 'scaling_cur_freq'), ('min', 'scaling_min_freq'), ('max', 'scaling_max_freq')]:
                freq_path = freq_base / filename
                if freq_path.exists():
                    freq_khz = int(freq_path.read_text().strip())
                    freq_info[key] = freq_khz / 1000
        except (IOError, ValueError, PermissionError):
            pass
        
        return freq_info
    
    def configure_environment(self) -> dict:
        """Set environment variables for optimized inference. Returns dict of vars set."""
        num_threads = self.get_optimal_num_threads()
        
        env_vars = {
            'OMP_NUM_THREADS': str(num_threads),
            'TF_NUM_INTEROP_THREADS': '1',
            'TF_NUM_INTRAOP_THREADS': str(num_threads),
            'OMP_WAIT_POLICY': 'PASSIVE',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return env_vars


class MemoryOptimizer:
    """Memory alignment for NEON SIMD operations (16-byte alignment for 128-bit registers)."""
    
    NEON_ALIGNMENT = 16
    CACHE_LINE_SIZE = 64
    
    @staticmethod
    def create_aligned_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32, alignment: int = 16) -> np.ndarray:
        """Create a numpy array with specified memory alignment."""
        dtype = np.dtype(dtype)
        total_size = int(np.prod(shape)) * dtype.itemsize
        
        buffer = np.empty(total_size + alignment, dtype=np.uint8)
        offset = alignment - (buffer.ctypes.data % alignment)
        if offset == alignment:
            offset = 0
        
        aligned_buffer = buffer[offset:offset + total_size]
        return aligned_buffer.view(dtype).reshape(shape)
    
    @staticmethod
    def is_aligned(array: np.ndarray, alignment: int = 16) -> bool:
        return array.ctypes.data % alignment == 0
    
    @staticmethod
    def ensure_contiguous(array: np.ndarray) -> np.ndarray:
        if array.flags['C_CONTIGUOUS']:
            return array
        return np.ascontiguousarray(array)
    
    @classmethod
    def prepare_input_buffer(cls, height: int, width: int, channels: int = 3, dtype: np.dtype = np.uint8) -> np.ndarray:
        """Pre-allocate aligned buffer for image input."""
        return cls.create_aligned_array((height, width, channels), dtype=dtype, alignment=cls.NEON_ALIGNMENT)


class QuantizationUtils:
    """Utilities for Int8/FP16 quantized models. Int8 is key for ≥5 FPS."""
    
    @staticmethod
    def calculate_scale_zero_point(min_val: float, max_val: float, num_bits: int = 8) -> Tuple[float, int]:
        """Calculate quantization scale and zero point for Int8."""
        qmin, qmax = 0, (1 << num_bits) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1e-8
        
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))
        
        return scale, zero_point
    
    @staticmethod
    def quantize_array(array: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Quantize float array to Int8."""
        quantized = np.round(array / scale + zero_point)
        return np.clip(quantized, 0, 255).astype(np.uint8)
    
    @staticmethod
    def dequantize_array(array: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize Int8 array to float."""
        return (array.astype(np.float32) - zero_point) * scale
    
    @staticmethod
    def normalize_image_uint8(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (127.5, 127.5, 127.5),
        std: Tuple[float, float, float] = (127.5, 127.5, 127.5)
    ) -> np.ndarray:
        """Normalize image for quantized model input. Maps [0,255] to [-1,1]."""
        image = image.astype(np.float32)
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        return (image - mean) / std
    
    @staticmethod
    def convert_to_fp16(array: np.ndarray) -> np.ndarray:
        return array.astype(np.float16)
    
    @staticmethod
    def convert_to_fp32(array: np.ndarray) -> np.ndarray:
        return array.astype(np.float32)


# Global instances (lazy initialization)
_platform_info: Optional[ArmPlatformInfo] = None
_thermal_monitor: Optional[ThermalMonitor] = None


def get_platform_info() -> ArmPlatformInfo:
    global _platform_info
    if _platform_info is None:
        _platform_info = ArmPlatformInfo()
    return _platform_info


def is_arm_platform() -> bool:
    return get_platform_info().is_arm


def is_raspberry_pi() -> bool:
    return get_platform_info().is_raspberry_pi


def get_optimal_num_threads() -> int:
    return CpuOptimizer(get_platform_info()).get_optimal_num_threads()


def create_aligned_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    return MemoryOptimizer.create_aligned_array(shape, dtype)


def get_cpu_temperature() -> float:
    global _thermal_monitor
    if _thermal_monitor is None:
        _thermal_monitor = ThermalMonitor()
    return _thermal_monitor.get_temperature()


def initialize_arm_optimizations() -> dict:
    """Initialize all Arm-specific optimizations. Call at application startup."""
    platform = get_platform_info()
    cpu_opt = CpuOptimizer(platform)
    
    env_vars = cpu_opt.configure_environment()
    
    thermal = ThermalMonitor()
    status, temp = thermal.get_status()
    
    return {
        'platform': platform.get_summary(),
        'num_threads': cpu_opt.get_optimal_num_threads(),
        'environment': env_vars,
        'thermal': {'status': status, 'temperature': temp}
    }


__all__ = [
    'ArmPlatformInfo', 'ThermalMonitor', 'CpuOptimizer', 'MemoryOptimizer', 'QuantizationUtils',
    'is_arm_platform', 'is_raspberry_pi', 'get_optimal_num_threads',
    'create_aligned_array', 'get_cpu_temperature', 'get_platform_info', 'initialize_arm_optimizations',
]


if __name__ == '__main__':
    print("=" * 60)
    print("KleidiAI Utils - Arm Optimization Module")
    print("=" * 60)
    
    config = initialize_arm_optimizations()
    
    print("\n📱 Platform Info:")
    for key, value in config['platform'].items():
        print(f"   {key}: {value}")
    
    print(f"\n⚡ Optimal Threads: {config['num_threads']}")
    print(f"\n🌡️  Thermal: {config['thermal']['status']} ({config['thermal']['temperature']}°C)")
    print("\n✅ Initialization complete!")
