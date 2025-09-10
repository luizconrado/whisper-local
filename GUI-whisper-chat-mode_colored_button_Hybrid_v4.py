#!/usr/bin/env python3
"""
Enhanced Hybrid Audio Transcriber Application - FINAL VERSION WITH SENSITIVITY CONFIG
====================================================================================
This version includes comprehensive fixes for all crash issues while maintaining
and enhancing all original functionality, PLUS dynamic sensitivity configuration.

Key Fixes Applied:
- Removed dangerous QThread.terminate() calls
- Added cooperative cancellation mechanism
- Fixed detected_language scoping bug
- Improved PyAudio resource management with singleton pattern
- Enhanced thread state tracking with registry
- Better signal connection management
- Added progress dialogs for user feedback
- Comprehensive error handling throughout

Enhanced Features:
- SENSITIVITY CONFIGURATION: 3 presets (Original/Balanced/Sensitive)
- Voice Activity Detection for intelligent chunking at natural pauses
- Advanced audio preprocessing pipeline (noise reduction, normalization, silence removal)
- 16kHz resampling optimized for MLX Whisper performance
- Real-time confidence indicators from Whisper transcription
- Single-step LLM refinement with audio quality context
- Optimized for moderate-to-low volume male voices
- Enhanced error handling and comprehensive logging
- Professional GUI with aligned confidence and quality indicators

All changes are marked with # FIX: and # SENSITIVITY: comments for traceability.

------------------------------------------------------------------------------------
ADDITIONAL NOTE (STEP-1 FIX APPLIED HERE):
- To prevent intermittent 'bus error' crashes, we disable word-level timestamps
  in MLX Whisper calls by setting 'word_timestamps': False. This reduces memory
  pressure across repeated transcribe() calls in a long-lived GUI.

- Because word-level probabilities may no longer be present, the confidence badge
  now falls back to segment-level avg_logprob (mapped to [0..1]) when available;
  otherwise it shows 'N/A' with a tooltip.
------------------------------------------------------------------------------------
"""

import sys
import re
import logging
import threading
import datetime
import tempfile
import wave
import numpy as np
import uuid
import time
import json
import os
import math  # FIX-STEP1: for isnan checks in confidence display
from dataclasses import dataclass
from functools import partial, wraps
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any, Type, Union
from enum import Enum
from contextvars import ContextVar

import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QMessageBox, QLabel,
    QProgressDialog
)
from PyQt5.QtCore import pyqtSignal, QThread, QObject, Qt

import mlx_whisper
import ollama

# Try to import additional libraries for audio processing
try:
    import scipy.signal as signal
    import scipy.io.wavfile as wavfile
    from scipy.ndimage import uniform_filter1d

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some audio preprocessing features will be limited.")

try:
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("webrtcvad not available. Using basic silence detection.")

# --------------------------
# Enhanced Logging Configuration
# --------------------------

# Context variable for correlation ID tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records for request tracking."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = correlation_id.get() or 'no-correlation'
        return True


class StructuredFormatter(logging.Formatter):
    """Format logs as structured JSON for better parsing."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'correlation_id': getattr(record, 'correlation_id', 'no-correlation')
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                           'levelname', 'levelno', 'lineno', 'module', 'msecs',
                           'message', 'pathname', 'process', 'processName', 'relativeCreated',
                           'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                           'correlation_id']:
                log_data[key] = value

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging with console output only.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Set log level
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with enhanced formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - '
        '%(correlation_id)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    console_handler.addFilter(CorrelationFilter())
    root_logger.addHandler(console_handler)

    # Also setup performance logger to use console
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = True  # Let it use root logger's console handler


def log_performance(func):
    """
    Decorator to log function performance metrics.

    Usage:
        @log_performance
        def my_function():
            # function code
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set correlation ID if not set
        if correlation_id.get() is None:
            correlation_id.set(str(uuid.uuid4()))

        perf_logger = logging.getLogger('performance')
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            perf_logger.info(
                f"Function {func.__name__} completed",
                extra={
                    'function': func.__name__,
                    'duration_ms': round(duration * 1000, 2),
                    'status': 'success'
                }
            )
            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            perf_logger.error(
                f"Function {func.__name__} failed",
                extra={
                    'function': func.__name__,
                    'duration_ms': round(duration * 1000, 2),
                    'status': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            )
            raise

    return wrapper


def set_correlation_id(new_id: Optional[str] = None) -> str:
    """
    Set a new correlation ID for the current context.

    Args:
        new_id: Optional correlation ID. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if new_id is None:
        new_id = str(uuid.uuid4())
    correlation_id.set(new_id)
    return new_id


# Initialize logging with default settings (console only)
setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))

# Create module logger
logger = logging.getLogger(__name__)
logger.info("Enhanced logging system initialized")


# --------------------------
# Custom Exception Classes
# --------------------------

class TranscriberError(Exception):
    """Base exception for all transcriber-specific errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception with message and optional details.

        Args:
            message: Error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.correlation_id = correlation_id.get()

        # Log the error
        logger.error(
            f"{self.__class__.__name__}: {message}",
            extra={'error_details': self.details, 'correlation_id': self.correlation_id}
        )


class AudioProcessingError(TranscriberError):
    """Raised when audio processing operations fail."""
    pass


class RecordingError(TranscriberError):
    """Raised when audio recording fails."""
    pass


class TranscriptionError(TranscriberError):
    """Raised when transcription fails."""
    pass


class RefinementError(TranscriberError):
    """Raised when text refinement fails."""
    pass


class ModelNotAvailableError(TranscriberError):
    """Raised when the requested model is not available."""
    pass


class ConfigurationError(TranscriberError):
    """Raised for configuration-related issues."""
    pass


class ThreadManagementError(TranscriberError):
    """Raised for thread management issues."""
    pass


class ResourceError(TranscriberError):
    """Raised when system resources are unavailable."""
    pass


class ValidationError(TranscriberError):
    """Raised when input validation fails."""
    pass


# --------------------------
# SENSITIVITY: Sensitivity Configuration System
# --------------------------

class SensitivityLevel(Enum):
    """Three sensitivity presets for different recording environments."""
    ORIGINAL = "original"  # Conservative - original settings
    BALANCED = "balanced"  # Medium - better for normal speech (current PATCHED)
    SENSITIVE = "sensitive"  # High - for very quiet speakers


@dataclass
class SensitivityConfig:
    """Configuration values for each sensitivity level."""

    # Audio Quality Thresholds
    quality_silence_poor: float
    quality_silence_fair: float
    quality_noise_high: float
    quality_noise_low: float
    quality_dynamic_excellent: float

    # Audio Processing Parameters
    silence_threshold: float
    silence_multiplier: float  # Multiplier for silence detection
    noise_percentile: int  # Percentile for noise floor estimation

    # Voice Amplification
    low_voice_amp: float
    male_voice_amp: float
    low_freq_boost: float
    male_freq_boost: float

    # Noise Reduction
    noise_reduce_strength: float

    # VAD Settings
    vad_aggressiveness: int  # 0-3, higher = more aggressive
    vad_energy_percentile: int  # Energy threshold percentile

    # Quality check adjustment
    min_rms_threshold: float  # Minimum RMS for "very quiet" detection


class GlobalAudioConfig:
    """
    Global configuration manager for audio sensitivity.
    Thread-safe singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._current_level = SensitivityLevel.BALANCED  # Default to current PATCHED behavior
        self._configs = self._create_configs()
        self._callbacks = []  # Callbacks for config changes

    def _create_configs(self) -> Dict[SensitivityLevel, SensitivityConfig]:
        """Create configuration presets for each sensitivity level."""

        return {
            SensitivityLevel.ORIGINAL: SensitivityConfig(
                # Original conservative thresholds (from original v4)
                quality_silence_poor=0.7,
                quality_silence_fair=0.5,
                quality_noise_high=0.1,
                quality_noise_low=0.05,
                quality_dynamic_excellent=0.3,

                # Original processing
                silence_threshold=0.003,
                silence_multiplier=1.0,  # No multiplier (original behavior)
                noise_percentile=10,  # Original percentile

                # Original amplification
                low_voice_amp=2.5,
                male_voice_amp=1.8,
                low_freq_boost=2.0,
                male_freq_boost=1.7,

                # Minimal noise reduction
                noise_reduce_strength=0.2,

                # Least aggressive VAD
                vad_aggressiveness=0,
                vad_energy_percentile=15,

                # Original threshold
                min_rms_threshold=0.01
            ),

            SensitivityLevel.BALANCED: SensitivityConfig(
                # Improved thresholds (current PATCHED values)
                quality_silence_poor=0.85,
                quality_silence_fair=0.7,
                quality_noise_high=0.2,
                quality_noise_low=0.1,
                quality_dynamic_excellent=0.2,

                # Better speech detection (current PATCHED)
                silence_threshold=0.003,
                silence_multiplier=3.0,  # Current PATCHED multiplier
                noise_percentile=5,  # Current PATCHED percentile

                # Same amplification as original
                low_voice_amp=2.5,
                male_voice_amp=1.8,
                low_freq_boost=2.0,
                male_freq_boost=1.7,

                # Moderate noise reduction
                noise_reduce_strength=0.3,

                # Balanced VAD
                vad_aggressiveness=0,  # Keep least aggressive
                vad_energy_percentile=15,

                # Adjusted threshold (current PATCHED)
                min_rms_threshold=0.008
            ),

            SensitivityLevel.SENSITIVE: SensitivityConfig(
                # Very lenient thresholds for quiet speech
                quality_silence_poor=0.9,  # Only almost complete silence is poor
                quality_silence_fair=0.8,  # Very high tolerance
                quality_noise_high=0.3,  # Less sensitive to noise
                quality_noise_low=0.15,  # More realistic for amplified audio
                quality_dynamic_excellent=0.15,  # Easier to achieve

                # Ultra-sensitive speech detection
                silence_threshold=0.001,  # Very sensitive
                silence_multiplier=5.0,  # Strong distinction
                noise_percentile=3,  # Lower percentile

                # Maximum amplification
                low_voice_amp=4.0,  # Higher amplification
                male_voice_amp=2.5,  # Stronger boost
                low_freq_boost=2.5,  # More bass boost
                male_freq_boost=2.0,  # More formant boost

                # Aggressive noise reduction to compensate
                noise_reduce_strength=0.5,  # Stronger noise reduction

                # More aggressive VAD
                vad_aggressiveness=1,  # Slightly more aggressive
                vad_energy_percentile=8,  # Lower threshold

                # Very low threshold
                min_rms_threshold=0.005  # Detect even quieter speech
            )
        }

    @property
    def current_level(self) -> SensitivityLevel:
        """Get current sensitivity level."""
        with self._lock:
            return self._current_level

    @current_level.setter
    def current_level(self, level: SensitivityLevel):
        """Set sensitivity level and notify callbacks."""
        with self._lock:
            if level != self._current_level:
                old_level = self._current_level
                self._current_level = level
                logger.info(f"Sensitivity changed from {old_level.value} to {level.value}")
                self._notify_callbacks()

    @property
    def config(self) -> SensitivityConfig:
        """Get current configuration."""
        with self._lock:
            return self._configs[self._current_level]

    def register_callback(self, callback):
        """Register a callback for configuration changes."""
        with self._lock:
            self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all callbacks of configuration change."""
        for callback in self._callbacks:
            try:
                callback(self._current_level, self.config)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")

    def get_config_value(self, attr_name: str, default=None):
        """Safely get a config value with fallback."""
        try:
            return getattr(self.config, attr_name, default)
        except Exception as e:
            logger.error(f"Error accessing config.{attr_name}: {e}")
            return default


# SENSITIVITY: Initialize global configuration singleton
GLOBAL_AUDIO_CONFIG = GlobalAudioConfig()


# --------------------------
# Audio Configuration
# --------------------------
class AudioConfig:
    """Enhanced audio configuration with preprocessing parameters.

    Attributes:
        SAMPLE_RATE: Recording sample rate (44100 Hz)
        WHISPER_SAMPLE_RATE: Whisper's optimal sample rate (16000 Hz)
        FORMAT: Audio format (16-bit PCM)
        CHANNELS: Number of audio channels (mono)
        CHUNK_SIZE: Buffer size for audio streaming

        Noise reduction parameters:
        NOISE_REDUCE_STRENGTH: Gentle noise reduction (0.2)
        NORMALIZE_TARGET_LEVEL: Target normalization level (-18.0 dB)
        SILENCE_THRESHOLD: Threshold for silence detection (0.003)
        MIN_SILENCE_DURATION: Minimum silence duration (0.4s)

        Voice optimization parameters (for moderate-to-low volume male voices):
        LOW_VOICE_AMPLIFICATION: Amplification for very quiet speech (2.5x)
        MALE_VOICE_AMPLIFICATION: Amplification for male voices (1.8x)
        LOW_FREQ_BOOST: Boost for low frequencies 80-250Hz (2.0x)
        MALE_FREQ_BOOST: Boost for male speech formants 300-800Hz (1.7x)

        VAD parameters:
        VAD_FRAME_DURATION: Frame duration for VAD analysis (30ms)
        VAD_AGGRESSIVENESS: VAD sensitivity level (0 = least aggressive)
    """
    SAMPLE_RATE = 44100  # Recording sample rate
    WHISPER_SAMPLE_RATE = 16000  # Whisper's optimal sample rate
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_SIZE = 1024

    # Audio preprocessing parameters (optimized for moderate-to-low male voice)
    NOISE_REDUCE_STRENGTH = 0.2  # Very gentle to preserve all speech nuances
    NORMALIZE_TARGET_LEVEL = -18.0  # dB - slightly higher for quiet speakers
    SILENCE_THRESHOLD = 0.003  # Very sensitive for consistently low volume
    MIN_SILENCE_DURATION = 0.4  # Slightly longer for natural speech patterns

    # Male voice optimization (moderate-to-low volume)
    LOW_VOICE_AMPLIFICATION = 2.5  # Higher amplification for consistent low volume
    MALE_VOICE_AMPLIFICATION = 1.8  # Additional boost specifically for male voices
    LOW_FREQ_BOOST = 2.0  # Stronger boost for male voice fundamentals (80-250Hz)
    MALE_FREQ_BOOST = 1.7  # Boost male speech formants (300-800Hz)

    # VAD parameters (optimized for soft-spoken male voice)
    VAD_FRAME_DURATION = 30  # ms
    VAD_AGGRESSIVENESS = 0  # Least aggressive - catches all quiet speech


# --------------------------
# Audio Quality Assessment
# --------------------------
class AudioQuality(Enum):
    """Audio quality levels for context in LLM prompts."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class AudioAnalysis:
    """Results of audio quality analysis."""
    quality: AudioQuality
    noise_level: float
    silence_ratio: float
    dynamic_range: float
    clipping_detected: bool
    duration: float


# --------------------------
# Audio Processing Constants
# --------------------------

# Quality thresholds for audio analysis
QUALITY_SILENCE_THRESHOLD_POOR = 0.85  # >70% silence = poor quality
QUALITY_SILENCE_THRESHOLD_FAIR = 0.7  # >50% silence = fair quality
QUALITY_NOISE_THRESHOLD_HIGH = 0.2  # >0.1 noise level = fair quality
QUALITY_NOISE_THRESHOLD_LOW = 0.1  # <0.05 noise level = excellent
QUALITY_DYNAMIC_RANGE_EXCELLENT = 0.2  # >0.3 dynamic range = excellent
QUALITY_CLIPPING_THRESHOLD = 0.95  # >0.95 amplitude = clipping

# Audio chunking parameters
CHUNK_MIN_DURATION_DEFAULT = 15.0  # Default minimum chunk duration (seconds)
CHUNK_MAX_DURATION_DEFAULT = 45.0  # Default maximum chunk duration (seconds)
CHUNK_MIN_DURATION_LONG = 15.0  # Min duration for long audio (>10 min)
CHUNK_MAX_DURATION_LONG = 35.0  # Max duration for long audio (>10 min)
CHUNK_OVERLAP_DURATION = 3.0  # Overlap between chunks (seconds)
CHUNK_SILENCE_GAP_THRESHOLD = 1.0  # Min silence gap for natural breaks (seconds)
CHUNK_MERGE_THRESHOLD = 15.0  # Min final chunk size to avoid merging

# Voice activity detection parameters
VAD_WINDOW_SIZE_MS = 25  # VAD analysis window size (milliseconds)
VAD_HOP_SIZE_MS = 10  # VAD hop size (milliseconds)
VAD_ENERGY_PERCENTILE = 15  # Percentile for energy threshold (low for quiet speech)
VAD_MAX_OVERLAP_WORDS = 10  # Max words to check for overlap removal

# Audio preprocessing parameters
AUDIO_WINDOW_SIZE_SEC = 0.1  # RMS calculation window size (seconds)
AUDIO_PERCENTILE_99 = 99  # High percentile for dynamic range
AUDIO_PERCENTILE_1 = 1  # Low percentile for dynamic range
AUDIO_PERCENTILE_10 = 10  # Noise level estimation percentile

# Male voice detection thresholds
MALE_VOICE_FREQ_RATIO = 0.7  # Low freq energy ratio for male voice detection
VERY_LOW_VOICE_FREQ_RATIO = 0.85  # Ratio for very deep/quiet voice detection

# Audio filtering frequency ranges (Hz)
FREQ_MALE_FUNDAMENTAL_LOW = 50  # Male fundamental frequency range start
FREQ_MALE_FUNDAMENTAL_HIGH = 250  # Male fundamental frequency range end
FREQ_MALE_FORMANT_LOW = 300  # Male formant frequency range start
FREQ_MALE_FORMANT_HIGH = 800  # Male formant frequency range end
FREQ_MALE_SPEECH_LOW = 80  # Male speech frequency range start
FREQ_MALE_SPEECH_HIGH = 800  # Male speech frequency range end

# Filter parameters
FILTER_ORDER_MINIMAL = 1  # Minimal filtering for very low voices
FILTER_ORDER_GENTLE = 2  # Gentle filtering for male voices
FILTER_ORDER_MODERATE = 3  # Moderate filtering for standard voices
FILTER_CUTOFF_VERY_LOW = 50  # Low cutoff for very deep voices
FILTER_CUTOFF_MALE = 55  # Low cutoff for male voices
FILTER_CUTOFF_STANDARD = 70  # Standard low cutoff
FILTER_CUTOFF_HIGH = 8000  # High frequency cutoff


# --------------------------
# Enhanced Model Configuration
# --------------------------
@dataclass
class ModelConfig:
    """Model configuration for single-step LLM text refinement."""
    name: str
    ctx_num: int
    temperature: float
    seed: int
    system_message: str
    user_message: str

    @classmethod
    def get_default_configs(cls):
        """Returns model configurations for single-step refinement."""
        return {
            'phi4:latest': cls(
                name='phi4:latest',
                ctx_num=8192,
                temperature=0.2,
                seed=1,
                system_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the '
                    'transcribed text, improve my vocabulary when necessary, making the text '
                    'clear and easy to understand. Also, add punctuation such as periods, '
                    'commas, and capitalization. Please use only the context provided. '
                    'As the output, I only want the corrected text, no preamble, '
                    'introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
                user_message='"{text}"'
            ),
            'deepseek-r1:1.5b': cls(
                name='deepseek-r1:1.5b',
                ctx_num=8192,
                temperature=1.3,
                seed=1,
                system_message="You are my helpful and professional text corrector.",
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                )
            ),
            'deepseek-r1:latest': cls(
                name='deepseek-r1:latest',
                ctx_num=8192,
                temperature=1.3,
                seed=1,
                system_message="You are my helpful and professional text corrector.",
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                )
            ),
            'deepseek-r1:14b': cls(
                name='deepseek-r1:14b',
                ctx_num=8192,
                temperature=1.3,
                seed=1,
                system_message="You are my helpful and professional text corrector.",
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                )
            ),
            'deepseek-r1:32b': cls(
                name='deepseek-r1:32b',
                ctx_num=4096,
                temperature=1.3,
                seed=1,
                system_message="You are my helpful and professional text corrector.",
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                )
            )
        }

    @classmethod
    def get_config(cls, model_name: str):
        """Retrieve configuration with fallback to default."""
        configs = cls.get_default_configs()
        if model_name in configs:
            return configs[model_name]
        else:
            logger.info(f"Model '{model_name}' not found. Using phi4:latest as default.")
            return configs['phi4:latest']


# --------------------------
# Enhanced Audio Processor
# --------------------------
class AudioProcessor:
    """Advanced audio preprocessing pipeline optimized for MLX Whisper transcription.

    Features:
    - Audio quality analysis and classification
    - Intelligent silence removal preserving speech
    - Noise reduction with male voice frequency preservation
    - Adaptive normalization for low-volume speakers
    - High-quality resampling to 16kHz for optimal Whisper performance
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.AudioProcessor")

    def analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> AudioAnalysis:
        """Analyze audio quality to provide context for LLM refinement."""
        # SENSITIVITY: Get dynamic configuration
        config = GLOBAL_AUDIO_CONFIG.config

        try:
            # Calculate basic metrics
            duration = len(audio_data) / sample_rate
            rms = np.sqrt(np.mean(audio_data ** 2))

            # SENSITIVITY: Detect silence regions using dynamic threshold
            silence_threshold = config.silence_threshold * config.silence_multiplier
            silence_mask = np.abs(audio_data) < silence_threshold
            silence_ratio = np.sum(silence_mask) / len(audio_data)

            # Calculate dynamic range (difference between loud and quiet parts)
            percentile_99 = np.percentile(np.abs(audio_data), AUDIO_PERCENTILE_99)
            percentile_1 = np.percentile(np.abs(audio_data), AUDIO_PERCENTILE_1)
            dynamic_range = percentile_99 - percentile_1

            # Detect clipping (audio exceeding maximum amplitude)
            max_value = np.max(np.abs(audio_data))  # Peak amplitude detection
            clipping_detected = max_value > QUALITY_CLIPPING_THRESHOLD

            # SENSITIVITY: Estimate noise floor using dynamic percentile
            noise_level = np.percentile(np.abs(audio_data), config.noise_percentile)

            # SENSITIVITY: Determine overall quality based on dynamic thresholds
            # Check for very low volume first
            is_very_quiet = rms < config.min_rms_threshold

            if clipping_detected or silence_ratio > config.quality_silence_poor:
                quality = AudioQuality.POOR
            elif is_very_quiet:
                quality = AudioQuality.FAIR  # Low volume gets fair, not poor
            elif (noise_level > config.quality_noise_high or
                  silence_ratio > config.quality_silence_fair):
                quality = AudioQuality.FAIR
            elif (dynamic_range > config.quality_dynamic_excellent and
                  noise_level < config.quality_noise_low and
                  silence_ratio < 0.5):  # Need <50% silence for excellent
                quality = AudioQuality.EXCELLENT
            else:
                quality = AudioQuality.GOOD

            return AudioAnalysis(
                quality=quality,
                noise_level=noise_level,
                silence_ratio=silence_ratio,
                dynamic_range=dynamic_range,
                clipping_detected=clipping_detected,
                duration=duration
            )
        except Exception as e:
            self.logger.error(f"Error analyzing audio quality: {e}")
            return AudioAnalysis(
                quality=AudioQuality.FAIR,
                noise_level=0.0,
                silence_ratio=0.0,
                dynamic_range=0.0,
                clipping_detected=False,
                duration=0.0
            )

    def resample_audio(self, audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_rate == target_rate:
            return audio_data

        if SCIPY_AVAILABLE:
            # High-quality resampling using scipy
            num_samples = int(len(audio_data) * target_rate / orig_rate)
            resampled = signal.resample(audio_data, num_samples)
            self.logger.info(f"Resampled audio from {orig_rate}Hz to {target_rate}Hz")
            return resampled.astype(np.float32)
        else:
            # Simple linear interpolation fallback
            ratio = target_rate / orig_rate
            indices = np.arange(0, len(audio_data), 1 / ratio)
            indices = indices[indices < len(audio_data)]
            resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            self.logger.info(f"Resampled audio using linear interpolation")
            return resampled.astype(np.float32)

    def remove_silence(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove long silent periods from audio."""
        try:
            # Calculate RMS energy in sliding windows (100ms windows with 50ms hop)
            # This allows detection of speech vs silence regions
            window_size = int(AUDIO_WINDOW_SIZE_SEC * sample_rate)  # 100ms windows
            hop_size = window_size // 2  # 50ms hop for overlap

            rms_values = []
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)

            # SENSITIVITY: Identify non-silent regions with dynamic threshold
            config = GLOBAL_AUDIO_CONFIG.config
            threshold = config.silence_threshold
            non_silent = np.array(rms_values) > threshold

            # Apply smoothing filter to avoid cutting speech at word boundaries
            if SCIPY_AVAILABLE:
                non_silent = uniform_filter1d(non_silent.astype(float), size=5) > 0.3

            # Create mask for original audio
            mask = np.zeros(len(audio_data), dtype=bool)
            for i, is_speech in enumerate(non_silent):
                start_idx = i * hop_size
                end_idx = min(start_idx + window_size, len(audio_data))
                if is_speech:
                    mask[start_idx:end_idx] = True

            cleaned_audio = audio_data[mask]

            # Handle case where all audio is removed
            if len(cleaned_audio) == 0:
                self.logger.warning("All audio was silence, keeping minimal audio")
                return audio_data[:min(1000, len(audio_data))]

            reduction_ratio = 1 - (len(cleaned_audio) / len(audio_data))
            self.logger.info(f"Removed {reduction_ratio:.1%} of audio as silence")

            return cleaned_audio
        except Exception as e:
            self.logger.error(f"Error removing silence: {e}")
            return audio_data

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to target level with low voice optimization."""
        try:
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms == 0:
                return audio_data

            # Detect voice characteristics for male moderate-to-low volume
            is_consistently_low = rms < 0.04
            is_very_low = rms < 0.015

            # SENSITIVITY: Apply appropriate amplification based on voice level
            config = GLOBAL_AUDIO_CONFIG.config
            if is_very_low:
                amplification = config.low_voice_amp
                audio_data = audio_data * amplification
                self.logger.info(f"Applied low voice amplification: {amplification}x")
            elif is_consistently_low:
                amplification = config.male_voice_amp
                audio_data = audio_data * amplification
                self.logger.info(f"Applied male voice amplification: {amplification}x")

            if is_consistently_low or is_very_low:
                rms = np.sqrt(np.mean(audio_data ** 2))

            if is_very_low:
                target_rms = 0.15
            elif is_consistently_low:
                target_rms = 0.13
            else:
                target_rms = 0.1
            scale_factor = target_rms / rms

            normalized = audio_data * scale_factor

            if np.any(np.isnan(normalized)):
                self.logger.warning("NaN detected in normalization, returning original")
                return audio_data

            normalized = np.clip(normalized, -0.95, 0.95)

            self.logger.info(f"Normalized audio with scale factor: {scale_factor:.3f}")
            return normalized
        except Exception as e:
            self.logger.error(f"Error normalizing audio: {e}")
            return audio_data

    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply basic noise reduction."""
        try:
            if not SCIPY_AVAILABLE:
                return audio_data

            # Apply gentle bandpass filter optimized for low voices
            noise_duration = min(int(0.5 * sample_rate), len(audio_data) // 4)
            _ = audio_data[:noise_duration]  # noise_sample not used explicitly

            nyquist = sample_rate / 2

            # Detect voice characteristics for male speech optimization
            low_freq_energy = np.mean(np.abs(audio_data[:int(0.1 * sample_rate)]))
            total_energy = np.mean(np.abs(audio_data))

            is_male_voice = low_freq_energy > total_energy * 0.7
            is_very_low_voice = low_freq_energy > total_energy * 0.85

            if is_very_low_voice:
                low_cutoff = 50
                high_cutoff = 8000
                order = 1
                self.logger.info("Detected very low/deep voice - minimal filtering")
            elif is_male_voice:
                low_cutoff = 55
                high_cutoff = 8000
                order = 2
                self.logger.info("Detected male voice - preserving low frequencies")
            else:
                low_cutoff = 70
                high_cutoff = 8000
                order = 3

            try:
                sos = signal.butter(order, [low_cutoff / nyquist, high_cutoff / nyquist],
                                    btype='band', output='sos')
                filtered = signal.sosfilt(sos, audio_data)

                config = GLOBAL_AUDIO_CONFIG.config
                if is_very_low_voice and config.low_freq_boost > 1.0:
                    fundamental_sos = signal.butter(2, [50 / nyquist, 250 / nyquist], btype='band', output='sos')
                    fundamental_freq = signal.sosfilt(fundamental_sos, filtered)
                    filtered = filtered + fundamental_freq * (config.low_freq_boost - 1.0)
                    self.logger.info(f"Applied fundamental frequency boost: {config.low_freq_boost}x")

                    formant_sos = signal.butter(2, [300 / nyquist, 800 / nyquist], btype='band', output='sos')
                    formant_freq = signal.sosfilt(formant_sos, filtered)
                    filtered = filtered + formant_freq * (config.male_freq_boost - 1.0)
                    self.logger.info(f"Applied male formant boost: {config.male_freq_boost}x")

                elif is_male_voice and config.male_freq_boost > 1.0:
                    male_voice_sos = signal.butter(2, [80 / nyquist, 800 / nyquist], btype='band', output='sos')
                    male_freq = signal.sosfilt(male_voice_sos, filtered)
                    filtered = filtered + male_freq * (config.male_freq_boost - 1.0)
                    self.logger.info(f"Applied male voice frequency boost: {config.male_freq_boost}x")

                filtered = np.clip(filtered, -0.95, 0.95)

                self.logger.info("Applied optimized bandpass filter")
                return filtered.astype(np.float32)
            except Exception as filter_error:
                self.logger.warning(f"Filter failed: {filter_error}, returning original")
                return audio_data

        except Exception as e:
            self.logger.error(f"Error reducing noise: {e}")
            return audio_data

    def process_audio(self, raw_audio: bytes) -> Tuple[np.ndarray, AudioAnalysis]:
        """Complete audio preprocessing pipeline."""
        try:
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            analysis = self.analyze_audio_quality(audio_np, AudioConfig.SAMPLE_RATE)
            self.logger.info(
                f"Audio analysis: {analysis.quality.value} quality, "
                f"{analysis.duration:.1f}s duration"
            )

            audio_np = self.remove_silence(audio_np, AudioConfig.SAMPLE_RATE)
            audio_np = self.reduce_noise(audio_np, AudioConfig.SAMPLE_RATE)
            audio_np = self.normalize_audio(audio_np)
            audio_np = self.resample_audio(
                audio_np, AudioConfig.SAMPLE_RATE, AudioConfig.WHISPER_SAMPLE_RATE
            )

            return audio_np, analysis

        except Exception as e:
            self.logger.error(f"Error in audio processing pipeline: {e}", exc_info=True)
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            analysis = AudioAnalysis(
                quality=AudioQuality.FAIR,
                noise_level=0.0,
                silence_ratio=0.0,
                dynamic_range=0.0,
                clipping_detected=False,
                duration=len(audio_np) / AudioConfig.SAMPLE_RATE
            )
            return audio_np, analysis


# --------------------------
# Voice Activity Detection
# --------------------------
class VoiceActivityDetector:
    """Intelligent voice activity detection for better chunking."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VAD")
        self.vad = None
        if VAD_AVAILABLE:
            config = GLOBAL_AUDIO_CONFIG.config
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            self.logger.info(f"WebRTC VAD initialized with aggressiveness {config.vad_aggressiveness}")
        else:
            self.logger.info("Using energy-based VAD fallback")

    def detect_speech_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Detect speech segments in audio data."""
        try:
            if self.vad and sample_rate == 16000:
                return self._webrtc_vad(audio_data, sample_rate)
            else:
                return self._energy_vad(audio_data, sample_rate)
        except Exception as e:
            self.logger.error(f"Error in VAD: {e}")
            duration = len(audio_data) / sample_rate
            return [(0.0, duration)]

    def _webrtc_vad(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Use WebRTC VAD for speech detection."""
        frame_duration_ms = AudioConfig.VAD_FRAME_DURATION
        frame_length = int(sample_rate * frame_duration_ms / 1000)

        audio_int16 = (audio_data * 32767).astype(np.int16)

        speech_segments = []
        current_segment_start = None

        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            frame_bytes = frame.tobytes()

            is_speech = self.vad.is_speech(frame_bytes, sample_rate)
            timestamp = i / sample_rate

            if is_speech and current_segment_start is None:
                current_segment_start = timestamp
            elif not is_speech and current_segment_start is not None:
                speech_segments.append((current_segment_start, timestamp))
                current_segment_start = None

        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio_data) / sample_rate))

        self.logger.info(f"WebRTC VAD detected {len(speech_segments)} speech segments")
        return speech_segments

    def _energy_vad(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Energy-based VAD fallback optimized for low-volume speech."""
        window_size = int(VAD_WINDOW_SIZE_MS / 1000.0 * sample_rate)  # 25ms
        hop_size = int(VAD_HOP_SIZE_MS / 1000.0 * sample_rate)  # 10ms

        energies = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            frame = audio_data[i:i + window_size]
            energy = np.sum(frame ** 2)
            energies.append(energy)

        energies = np.array(energies)

        config = GLOBAL_AUDIO_CONFIG.config
        threshold = np.percentile(energies, config.vad_energy_percentile)

        is_speech = energies > threshold
        speech_segments = []
        current_start = None

        for i, speech in enumerate(is_speech):
            timestamp = i * hop_size / sample_rate

            if speech and current_start is None:
                current_start = timestamp
            elif not speech and current_start is not None:
                speech_segments.append((current_start, timestamp))
                current_start = None

        if current_start is not None:
            speech_segments.append((current_start, len(audio_data) / sample_rate))

        self.logger.info(f"Energy VAD detected {len(speech_segments)} speech segments")
        return speech_segments

    def get_optimal_chunks(self, audio_data: np.ndarray, sample_rate: int,
                           min_chunk_duration: float = 15.0, max_chunk_duration: float = 45.0) -> List[
        Tuple[float, float]]:
        """Get optimal chunk boundaries with minimum duration and natural pause detection."""
        try:
            speech_segments = self.detect_speech_segments(audio_data, sample_rate)

            if not speech_segments:
                duration = len(audio_data) / sample_rate
                chunks = []
                for start in np.arange(0, duration, min_chunk_duration):
                    end = min(start + max_chunk_duration, duration)
                    chunks.append((start, end))
                return chunks

            chunks = []
            current_chunk_start = 0.0
            last_silence_end = 0.0

            for segment_start, segment_end in speech_segments:
                current_duration = segment_end - current_chunk_start

                if current_duration >= min_chunk_duration:
                    silence_gap = segment_start - last_silence_end

                    if (current_duration >= max_chunk_duration or
                            (silence_gap > CHUNK_SILENCE_GAP_THRESHOLD and current_duration > min_chunk_duration)):
                        chunks.append((current_chunk_start, last_silence_end))
                        current_chunk_start = segment_start

                last_silence_end = segment_end

            total_duration = len(audio_data) / sample_rate
            if current_chunk_start < total_duration:
                final_duration = total_duration - current_chunk_start

                if final_duration < CHUNK_MERGE_THRESHOLD and chunks:
                    last_start, _ = chunks.pop()
                    chunks.append((last_start, total_duration))
                else:
                    chunks.append((current_chunk_start, total_duration))

            self.logger.info(
                f"Created {len(chunks)} optimal chunks (min: {min_chunk_duration}s, max: {max_chunk_duration}s)")
            return chunks

        except Exception as e:
            self.logger.error(f"Error creating optimal chunks: {e}")
            duration = len(audio_data) / sample_rate
            chunks = []
            for start in np.arange(0, duration, min_chunk_duration):
                end = min(start + max_chunk_duration, duration)
                chunks.append((start, end))
            return chunks


# --------------------------
# FIX: Enhanced Application State with Robust Thread Tracking
# --------------------------
class AppState(QObject):
    """Enhanced application state with robust thread tracking and audio processing context."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._is_recording = False
        self.audio_buffer = bytes()
        self.active_threads = 0
        self._thread_registry = {}  # FIX: Track threads by ID with metadata
        self.audio_processor = AudioProcessor()
        self.vad = VoiceActivityDetector()
        self.last_audio_analysis: Optional[AudioAnalysis] = None

    @property
    def is_recording(self):
        with self._lock:
            return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool):
        with self._lock:
            self._is_recording = value

    def register_thread(self, thread_id: str):
        """Register a new active thread with metadata."""
        with self._lock:
            self._thread_registry[thread_id] = {
                'start_time': datetime.datetime.now(),
                'status': 'active'
            }
            self.active_threads = len([t for t in self._thread_registry.values()
                                       if t['status'] == 'active'])
            logger.info(f"Thread {thread_id} registered. Active threads: {self.active_threads}")

    def unregister_thread(self, thread_id: str):
        """Unregister a thread (completed or failed)."""
        with self._lock:
            if thread_id in self._thread_registry:
                self._thread_registry[thread_id]['status'] = 'completed'
                self._thread_registry[thread_id]['end_time'] = datetime.datetime.now()
            self.active_threads = len([t for t in self._thread_registry.values()
                                       if t['status'] == 'active'])
            logger.info(f"Thread {thread_id} unregistered. Active threads: {self.active_threads}")

    def cleanup_stale_threads(self, timeout_seconds: int = 300):
        """Clean up threads that have been running too long."""
        with self._lock:
            now = datetime.datetime.now()
            for thread_id, info in list(self._thread_registry.items()):
                if info['status'] == 'active':
                    runtime = (now - info['start_time']).total_seconds()
                    if runtime > timeout_seconds:
                        logger.warning(f"Thread {thread_id} timed out after {runtime:.1f}s")
                        info['status'] = 'timeout'
            self.active_threads = len([t for t in self._thread_registry.values()
                                       if t['status'] == 'active'])

    @property
    def has_active_threads(self):
        with self._lock:
            return self.active_threads > 0

    def set_audio_analysis(self, analysis: AudioAnalysis):
        with self._lock:
            self.last_audio_analysis = analysis

    def get_audio_analysis(self) -> Optional[AudioAnalysis]:
        with self._lock:
            return self.last_audio_analysis


# --------------------------
# FIX: Enhanced Audio Recorder with Better Resource Management
# --------------------------
class AudioRecorder:
    """Audio recorder with improved resource management using singleton pattern."""

    _pyaudio_instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_pyaudio(cls):
        """Get or create singleton PyAudio instance."""
        with cls._instance_lock:
            if cls._pyaudio_instance is None:
                try:
                    cls._pyaudio_instance = pyaudio.PyAudio()
                    logger.info("Created singleton PyAudio instance")
                except Exception as e:
                    raise ResourceError(
                        "Could not initialize PyAudio",
                        details={'original_error': str(e), 'error_type': type(e).__name__}
                    ) from e
            return cls._pyaudio_instance

    def __init__(self, state: AppState):
        self.state = state
        self.audio = None
        self.stream = None
        self._cleanup_done = False

    def __enter__(self):
        try:
            self.audio = self.get_pyaudio()

            device_count = self.audio.get_device_count()
            if device_count == 0:
                raise ResourceError(
                    "No audio devices found",
                    details={'device_count': 0}
                )

            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                frames_per_buffer=AudioConfig.CHUNK_SIZE,
                stream_callback=None,
                input_device_index=None
            )
            logger.info("Audio stream opened successfully")
            return self
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            self._safe_cleanup()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self._safe_cleanup()

    def _safe_cleanup(self):
        """Safely cleanup audio resources."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        try:
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream closed")
        except Exception as e:
            logger.error(f"Error closing stream: {e}")

        self.audio = None

    def record(self):
        """Record audio with improved error handling."""
        if not self.stream:
            logger.error("Stream not initialized")
            return None

        recorded_data = []
        start_time = datetime.datetime.now()
        try:
            while self.state.is_recording:
                try:
                    data = self.stream.read(AudioConfig.CHUNK_SIZE,
                                            exception_on_overflow=False)
                    recorded_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading audio chunk: {e}")
                    if not self.stream.is_active():
                        logger.error("Stream became inactive during recording")
                        break

            audio_bytes = b"".join(recorded_data)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Recorded {len(audio_bytes)} bytes of audio in {duration:.1f}s")
            return audio_bytes

        except Exception as e:
            logger.error(f"Recording failed: {e}", exc_info=True)
            return None


# --------------------------
# FIX: Enhanced Transcription Thread with Cancellation Support
# --------------------------
class TranscriptionThread(QThread):
    """Enhanced transcription with advanced preprocessing, VAD chunking, and cancellation support."""
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)
    confidence_updated = pyqtSignal(float)  # Signal for confidence updates
    error_occurred = pyqtSignal(str)

    def __init__(self, audio_data: bytes, model_name: str, state: AppState):
        super().__init__()
        self.audio_data = audio_data
        self.model_name = model_name
        self.state = state
        self._thread_id = f"transcribe_{id(self)}_{datetime.datetime.now().timestamp()}"
        self._is_cancelled = False
        self._cancel_lock = threading.Lock()

    def cancel(self):
        """Request cancellation of this thread."""
        with self._cancel_lock:
            self._is_cancelled = True
            logger.info(f"Cancellation requested for thread {self._thread_id}")

    def is_cancelled(self):
        """Check if cancellation was requested."""
        with self._cancel_lock:
            return self._is_cancelled

    def run(self) -> None:
        """Enhanced processing with reliable state tracking and cancellation support."""
        self.state.register_thread(self._thread_id)

        try:
            if not self.audio_data:
                self.error_occurred.emit("No audio data to transcribe.")
                return

            if self.is_cancelled():
                logger.info(f"Thread {self._thread_id} cancelled before processing")
                return

            transcription, confidence_info = self.transcribe_audio_enhanced()

            if not self.is_cancelled():
                self.transcription_finished.emit(transcription)

                if 'avg_confidence' in confidence_info:
                    # May be NaN when not computable; UI handles it as "N/A"
                    self.confidence_updated.emit(float(confidence_info['avg_confidence']))

                if "Failed to transcribe" not in transcription and "Transcription resulted in no text" not in transcription:
                    refined = self.refine_text(transcription, confidence_info)
                    if not self.is_cancelled():
                        self.refinement_finished.emit(refined)
                else:
                    self.refinement_finished.emit("")

        except Exception as e:
            logger.exception(f"Error in TranscriptionThread {self._thread_id}")
            self.error_occurred.emit(str(e))
        finally:
            self.state.unregister_thread(self._thread_id)

    def transcribe_audio_enhanced(self) -> Tuple[str, Dict[str, Any]]:
        """Enhanced transcription with preprocessing, VAD chunking, and confidence scoring."""
        try:
            processed_audio, audio_analysis = self.state.audio_processor.process_audio(self.audio_data)
            self.state.set_audio_analysis(audio_analysis)

            logger.info(f"Processing {audio_analysis.duration:.1f}s of {audio_analysis.quality.value} quality audio")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                audio_int16 = (processed_audio * 32767).astype(np.int16)

                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(AudioConfig.CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(AudioConfig.WHISPER_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())

                total_duration = len(processed_audio) / AudioConfig.WHISPER_SAMPLE_RATE

                if total_duration > 600:
                    min_chunk, max_chunk = CHUNK_MIN_DURATION_LONG, CHUNK_MAX_DURATION_LONG
                else:
                    min_chunk, max_chunk = CHUNK_MIN_DURATION_DEFAULT, CHUNK_MAX_DURATION_DEFAULT

                chunks = self.state.vad.get_optimal_chunks(
                    processed_audio, AudioConfig.WHISPER_SAMPLE_RATE,
                    min_chunk_duration=min_chunk, max_chunk_duration=max_chunk
                )

                if len(chunks) == 1 and chunks[0][1] - chunks[0][0] <= 30:
                    if self.is_cancelled():
                        return "Transcription cancelled.", {}

                    detected_language = self._detect_language_fast(temp_wav.name)

                    mlx_params = self._get_mlx_params_for_quality(audio_analysis)

                    result = mlx_whisper.transcribe(
                        temp_wav.name,
                        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                        initial_prompt=self._get_enhanced_prompt(audio_analysis),
                        language=detected_language,
                        **mlx_params
                    )

                    confidence_info = {
                        'avg_confidence': self._calculate_avg_confidence(result.get('segments', [])),
                        'low_confidence_words': self._get_low_confidence_words(result.get('segments', [])),
                        'audio_quality': audio_analysis.quality.value
                    }

                    return result['text'].strip(), confidence_info
                else:
                    return self._transcribe_chunks_enhanced(temp_wav.name, chunks, processed_audio, audio_analysis)

        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}", exc_info=True)
            return f"Failed to transcribe: {str(e)}", {}

    def _transcribe_chunks_enhanced(self, wav_path: str, chunks: List[Tuple[float, float]],
                                    audio_data: np.ndarray, audio_analysis: AudioAnalysis) -> Tuple[
        str, Dict[str, Any]]:
        """Transcribe audio chunks with overlap, confidence tracking, and cancellation support."""
        try:
            full_transcription = []
            all_confidences = []
            low_confidence_words = []
            previous_text = ""

            detected_language = 'en'
            language_detected = False

            overlap_duration = CHUNK_OVERLAP_DURATION
            sample_rate = AudioConfig.WHISPER_SAMPLE_RATE

            for i, (start_time, end_time) in enumerate(chunks):
                if self.is_cancelled():
                    logger.info(f"Transcription cancelled at chunk {i + 1}/{len(chunks)}")
                    return "Transcription cancelled by user.", {}

                logger.info(f"Processing chunk {i + 1}/{len(chunks)}: {start_time:.1f}s - {end_time:.1f}s")

                actual_start = max(0, start_time - (overlap_duration if i > 0 else 0))
                start_sample = int(actual_start * sample_rate)
                end_sample = int(end_time * sample_rate)

                chunk_audio = audio_data[start_sample:end_sample]

                with tempfile.NamedTemporaryFile(suffix=f'_chunk_{i}.wav', delete=True) as chunk_file:
                    chunk_int16 = (chunk_audio * 32767).astype(np.int16)

                    with wave.open(chunk_file.name, 'wb') as wf:
                        wf.setnchannels(AudioConfig.CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(chunk_int16.tobytes())

                    try:
                        if i == 0 and not language_detected:
                            try:
                                detected_language = self._detect_language_fast(chunk_file.name)
                                language_detected = True
                                logger.info(f"Detected language: {detected_language}")
                            except Exception as lang_error:
                                logger.warning(f"Language detection failed: {lang_error}, using 'en'")
                                detected_language = 'en'
                                language_detected = True

                        context_prompt = self._get_enhanced_prompt(audio_analysis)
                        if previous_text:
                            context_prompt += f" Previous context: {previous_text}"

                        mlx_params = self._get_mlx_params_for_quality(audio_analysis)

                        result = mlx_whisper.transcribe(
                            chunk_file.name,
                            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                            initial_prompt=context_prompt,
                            language=detected_language,
                            **mlx_params
                        )

                        chunk_text = result['text'].strip()

                        if i > 0 and overlap_duration > 0:
                            chunk_text = self._remove_overlap_text(chunk_text, previous_text)

                        full_transcription.append(chunk_text)

                        if full_transcription:
                            all_previous = " ".join(full_transcription)
                            previous_text = all_previous[-200:] if len(all_previous) > 200 else all_previous

                        segments = result.get('segments', [])
                        chunk_confidence = self._calculate_avg_confidence(segments)
                        all_confidences.append(chunk_confidence)

                        low_confidence_words.extend(self._get_low_confidence_words(segments))

                        logger.info(
                            f"Chunk {i + 1} transcribed with {chunk_confidence if not math.isnan(chunk_confidence) else 'N/A'} confidence")

                    except Exception as e:
                        logger.error(f"Failed to transcribe chunk {i + 1}: {e}")
                        full_transcription.append(f"[Error in chunk {i + 1}]")
                        all_confidences.append(float('nan'))

            # Combine results
            valid_confidences = [c for c in all_confidences if not math.isnan(c)]
            avg_confidence = (np.mean(valid_confidences) if valid_confidences else float('nan'))

            confidence_info = {
                'avg_confidence': avg_confidence,
                'low_confidence_words': low_confidence_words,
                'audio_quality': audio_analysis.quality.value,
                'language': detected_language
            }

            final_text = " ".join(full_transcription).strip()
            logger.info(
                f"Final transcription completed with {avg_confidence if not math.isnan(avg_confidence) else 'N/A'} average confidence")
            return final_text, confidence_info

        except Exception as e:
            logger.error(f"Error in enhanced chunk transcription: {e}")
            return f"Failed to transcribe chunks: {str(e)}", {}

    def _get_enhanced_prompt(self, audio_analysis: AudioAnalysis) -> str:
        """Generate context-aware prompt based on audio quality."""
        base_prompt = "Accurate transcript with proper punctuation and capitalization."

        if audio_analysis.quality == AudioQuality.POOR:
            return f"{base_prompt} Audio quality is poor with possible noise or distortion."
        elif audio_analysis.quality == AudioQuality.FAIR:
            return f"{base_prompt} Audio has some background noise."
        elif audio_analysis.clipping_detected:
            return f"{base_prompt} Audio may have some clipping or distortion."
        else:
            return f"{base_prompt} Clear audio recording."

    def _get_mlx_params_for_quality(self, audio_analysis: AudioAnalysis) -> Dict[str, Any]:
        """Get MLX Whisper parameters optimized for audio quality."""
        base_params = {
            # FIX-STEP1: disable word-level timestamps to prevent memory pressure across calls
            'word_timestamps': False,
            'condition_on_previous_text': True,
            'prepend_punctuations': '"\'-([{-',
            'append_punctuations': '"\'.!?:)]}',
            'hallucination_silence_threshold': 2.0
        }

        if audio_analysis.quality == AudioQuality.POOR:
            return {
                **base_params,
                'temperature': (0.0, 0.2, 0.4, 0.6, 0.8),
                'compression_ratio_threshold': 3.0,
                'logprob_threshold': -1.5,
                'no_speech_threshold': 0.4,
            }
        elif audio_analysis.quality == AudioQuality.FAIR:
            return {
                **base_params,
                'temperature': (0.0, 0.2, 0.4, 0.6),
                'compression_ratio_threshold': 2.8,
                'logprob_threshold': -1.2,
                'no_speech_threshold': 0.5,
            }
        elif audio_analysis.quality == AudioQuality.GOOD:
            return {
                **base_params,
                'temperature': (0.0, 0.2, 0.4),
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
            }
        else:  # EXCELLENT
            return {
                **base_params,
                'temperature': 0.0,
                'compression_ratio_threshold': 2.0,
                'logprob_threshold': -0.5,
                'no_speech_threshold': 0.7,
            }

    def _detect_language_fast(self, audio_chunk_path: str) -> str:
        """Fast language detection using a smaller model."""
        try:
            result = mlx_whisper.transcribe(
                audio_chunk_path,
                path_or_hf_repo="mlx-community/whisper-tiny",
                verbose=False
            )
            detected_lang = result.get('language', 'en')
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to 'en'")
            return 'en'

    def _calculate_avg_confidence(self, segments: List[Dict]) -> float:
        """Return confidence in [0,1] when possible; NaN when not computable.

        Primary (old path): average of word-level 'probability' if present.
        Fallback: map segment-level 'avg_logprob' (≈ [-5, 0]) to [0,1] via logistic.
        """
        if not segments:
            return float('nan')

        # Path 1: word-level probabilities (present only if word_timestamps were produced)
        confidences = []
        for segment in segments:
            words = segment.get('words', [])
            for word in words:
                prob = word.get('probability')
                if prob is not None:
                    confidences.append(prob)

        if confidences:
            return float(np.mean(confidences))

        # Path 2 (fallback): segment avg_logprob -> [0..1] proxy via logistic mapping
        # Typical avg_logprob for Whisper lies roughly in [-5, 0]; shift and scale.
        proxies = []
        for segment in segments:
            lp = segment.get('avg_logprob')
            if lp is not None:
                proxies.append(1.0 / (1.0 + np.exp(-2.0 * (lp + 1.0))))  # tunable

        if proxies:
            return float(np.mean(proxies))

        # Not computable -> NaN (UI will display "N/A")
        return float('nan')

    def _get_low_confidence_words(self, segments: List[Dict], threshold: float = 0.5) -> List[str]:
        """Extract low-confidence words when available (word timestamps path)."""
        low_conf_words = []
        for segment in segments:
            words = segment.get('words', [])
            for word in words:
                if 'probability' in word and word['probability'] < threshold:
                    low_conf_words.append(word.get('word', '').strip())
        return low_conf_words

    def _remove_overlap_text(self, current_text: str, previous_text: str) -> str:
        """Remove overlapping text between chunks."""
        if not previous_text:
            return current_text

        current_words = current_text.split()
        previous_words = previous_text.split()

        max_overlap = min(VAD_MAX_OVERLAP_WORDS, len(current_words), len(previous_words))

        for overlap_len in range(max_overlap, 0, -1):
            if current_words[:overlap_len] == previous_words[-overlap_len:]:
                return " ".join(current_words[overlap_len:])

        return current_text

    def refine_text(self, text: str, confidence_info: Dict[str, Any] = None) -> str:
        """Single-step text refinement with optional audio quality context."""
        try:
            if self.is_cancelled():
                return "Refinement cancelled."

            config = ModelConfig.get_config(self.model_name)
            audio_analysis = self.state.get_audio_analysis()

            enhanced_system = config.system_message
            if audio_analysis and confidence_info:
                context = f"\n\nContext: This text was transcribed from {audio_analysis.quality.value} quality audio"
                if confidence_info.get('low_confidence_words'):
                    context += f" with some uncertain words: {', '.join(confidence_info['low_confidence_words'][:5])}"
                enhanced_system += context

            messages = [
                {'role': 'system', 'content': enhanced_system},
                {'role': 'user', 'content': config.user_message.format(text=text)}
            ]

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'ctx_num': config.ctx_num,
                    'temperature': config.temperature,
                    'seed': config.seed
                }
            )

            refined = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)
            result = refined.strip().strip('"')

            logger.info("Completed single-step text refinement")
            return result

        except Exception as e:
            logger.error(f"Text refinement failed: {e}")
            return f"Refinement failed: {str(e)}"


# --------------------------
# FIX: Enhanced Refinement Thread with Cancellation
# --------------------------
class RefinementThread(QThread):
    """Single-step refinement thread with cancellation support."""
    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, text: str, model_name: str, state: AppState):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.state = state
        self._thread_id = f"refine_{id(self)}_{datetime.datetime.now().timestamp()}"
        self._is_cancelled = False
        self._cancel_lock = threading.Lock()

    def cancel(self):
        """Request cancellation of this thread."""
        with self._cancel_lock:
            self._is_cancelled = True
            logger.info(f"Cancellation requested for refinement thread {self._thread_id}")

    def is_cancelled(self):
        """Check if cancellation was requested."""
        with self._cancel_lock:
            return self._is_cancelled

    def run(self) -> None:
        """Single-step refinement process with tracking."""
        self.state.register_thread(self._thread_id)

        try:
            if self.is_cancelled():
                logger.info(f"Refinement thread {self._thread_id} cancelled before processing")
                return

            audio_analysis = self.state.get_audio_analysis()
            confidence_info = {'low_confidence_words': [], 'audio_quality': 'unknown'}

            if audio_analysis:
                confidence_info['audio_quality'] = audio_analysis.quality.value

            refined = self.refine_text(self.text, confidence_info)

            if not self.is_cancelled():
                self.refinement_finished.emit(refined)

        except Exception as e:
            logger.error(f"Error in RefinementThread {self._thread_id}", exc_info=True)
            self.error_occurred.emit(str(e))
        finally:
            self.state.unregister_thread(self._thread_id)

    def refine_text(self, text: str, confidence_info: Dict[str, Any] = None) -> str:
        """Single-step text refinement with optional audio quality context."""
        try:
            if self.is_cancelled():
                return "Refinement cancelled."

            config = ModelConfig.get_config(self.model_name)
            audio_analysis = self.state.get_audio_analysis()

            enhanced_system = config.system_message
            if audio_analysis and confidence_info:
                context = f"\n\nContext: This text was transcribed from {audio_analysis.quality.value} quality audio"
                if confidence_info.get('low_confidence_words'):
                    context += f" with some uncertain words: {', '.join(confidence_info['low_confidence_words'][:5])}"
                enhanced_system += context

            messages = [
                {'role': 'system', 'content': enhanced_system},
                {'role': 'user', 'content': config.user_message.format(text=text)}
            ]

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'ctx_num': config.ctx_num,
                    'temperature': config.temperature,
                    'seed': config.seed
                }
            )

            refined = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)
            result = refined.strip().strip('"')

            logger.info("Completed single-step text refinement")
            return result

        except Exception as e:
            logger.error(f"Text refinement failed: {e}")
            return f"Refinement failed: {str(e)}"


# --------------------------
# FIX: Enhanced Main GUI Application with Safe Thread Management
# --------------------------
class AudioTranscriberApp(QWidget):
    """Enhanced GUI application maintaining original layout and functionality with safe thread management."""

    _transcription_ready = pyqtSignal(bytes)
    _recording_failed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.available_models = self.fetch_models()
        self.current_transcription = ""
        self.current_worker = None
        self._worker_connections = []

        self._transcription_ready.connect(self._handle_transcription_ready)
        self._recording_failed.connect(self._handle_recording_failed)

        self.init_ui()

    def _handle_transcription_ready(self, audio_data: bytes):
        """Handle transcription ready signal (runs on main thread)."""
        self.current_transcription = ""
        self.start_transcription(audio_data)

    def _handle_recording_failed(self):
        """Handle recording failed signal (runs on main thread)."""
        self.display_transcription("No audio data captured.")
        self.display_refined_text("")

    def _disconnect_worker_signals(self):
        """Safely disconnect all worker signals."""
        for connection in self._worker_connections:
            try:
                connection[0].disconnect(connection[1])
                logger.debug(f"Disconnected signal: {connection[0]}")
            except (TypeError, RuntimeError) as e:
                logger.debug(f"Signal already disconnected: {e}")
        self._worker_connections.clear()

    def _connect_worker_signals(self, worker):
        """Connect worker signals and track connections."""
        self._disconnect_worker_signals()

        connections = [
            (worker.transcription_finished, self.display_transcription),
            (worker.refinement_finished, self.display_refined_text),
            (worker.confidence_updated, self.update_confidence_display),
            (worker.error_occurred, self.handle_error)
        ]

        for signal, slot in connections:
            signal.connect(slot)
            self._worker_connections.append((signal, slot))
            logger.debug(f"Connected signal: {signal}")

    def fetch_models(self) -> List[str]:
        """Fetch available models with same logic as original."""
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]
            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')
            return installed_models if installed_models else ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logger.error("Failed to fetch models", exc_info=True)
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def init_ui(self) -> None:
        """Initialize UI with same layout as original plus confidence display."""
        self.setWindowTitle("Enhanced Hybrid Audio Transcriber - COMPLETE FIXED")
        self.setGeometry(420, 300, 800, 500)
        main_layout = QVBoxLayout(self)
        self.create_top_controls(main_layout)
        self.create_progress_bar(main_layout)
        self.create_text_areas(main_layout)
        self.show()

    def create_top_controls(self, layout) -> None:
        """Create top controls with same styling as original."""
        top_layout = QHBoxLayout()

        self.recording_button = QPushButton("Start Recording", self)
        self.set_button_style("ready")
        self.recording_button.clicked.connect(self.toggle_recording)
        top_layout.addWidget(self.recording_button, 50)

        sensitivity_label = QLabel("Sensitivity:", self)
        sensitivity_label.setStyleSheet("color: white; font-weight: bold; padding: 0 8px;")
        top_layout.addWidget(sensitivity_label)

        self.sensitivity_selector = QComboBox(self)
        self.sensitivity_selector.addItems(["Original", "Balanced", "Sensitive"])
        self.sensitivity_selector.setCurrentIndex(1)
        self.sensitivity_selector.setStyleSheet("""
            QComboBox {
                padding: 8px;
                min-width: 100px;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #34495e;
                color: white;
            }
            QComboBox::drop-down { width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        self.sensitivity_selector.currentTextChanged.connect(self.on_sensitivity_changed)
        top_layout.addWidget(self.sensitivity_selector, 15)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.available_models)
        if 'phi4:latest' in self.available_models:
            self.model_selector.setCurrentText('phi4:latest')
        else:
            self.model_selector.setCurrentIndex(0)
        self.model_selector.setStyleSheet("""
            QComboBox { 
                padding: 8px; 
                min-width: 120px; 
                font-size: 14px; 
                border: 1px solid #cccccc; 
                border-radius: 4px; 
            }
            QComboBox::drop-down { width: 20px; }
        """)
        top_layout.addWidget(self.model_selector, 30)

        self.re_refine_button = QPushButton("Re-Refine Text", self)
        self.re_refine_button.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #34495e; }
        """)
        self.re_refine_button.clicked.connect(self.re_refine_text)
        top_layout.addWidget(self.re_refine_button, 20)

        layout.addLayout(top_layout)

    def create_progress_bar(self, layout) -> None:
        """Create progress bar (same as original)."""
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def create_text_areas(self, layout) -> None:
        """Create text areas with aligned confidence indicators."""
        text_layout = QHBoxLayout()

        trans_layout = QVBoxLayout()
        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Original transcription...")
        trans_layout.addWidget(self.transcription_box)

        trans_confidence_layout = QHBoxLayout()
        confidence_text = QLabel("Transcription Confidence:", self)
        confidence_text.setStyleSheet("font-weight: bold; color: white;")
        trans_confidence_layout.addWidget(confidence_text)

        self.confidence_label = QLabel("--", self)
        self.confidence_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 60px;
            }
        """)
        trans_confidence_layout.addWidget(self.confidence_label)
        trans_confidence_layout.addStretch()
        trans_layout.addLayout(trans_confidence_layout)

        self.copy_transcription_btn = QPushButton("Copy", self)
        self.copy_transcription_btn.clicked.connect(partial(self.copy_text, self.transcription_box))
        trans_layout.addWidget(self.copy_transcription_btn)
        text_layout.addLayout(trans_layout)

        refined_layout = QVBoxLayout()
        self.refined_box = QTextEdit(self)
        self.refined_box.setPlaceholderText("Enhanced refined text...")
        refined_layout.addWidget(self.refined_box)

        quality_confidence_layout = QHBoxLayout()
        quality_text = QLabel("Audio Quality:", self)
        quality_text.setStyleSheet("font-weight: bold; color: white;")
        quality_confidence_layout.addWidget(quality_text)

        self.quality_label = QLabel("--", self)
        self.quality_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 80px;
            }
        """)
        quality_confidence_layout.addWidget(self.quality_label)
        quality_confidence_layout.addStretch()
        refined_layout.addLayout(quality_confidence_layout)

        self.copy_refined_btn = QPushButton("Copy", self)
        self.copy_refined_btn.clicked.connect(partial(self.copy_text, self.refined_box))
        refined_layout.addWidget(self.copy_refined_btn)
        text_layout.addLayout(refined_layout)

        layout.addLayout(text_layout)

    def set_button_style(self, state):
        """Set button style (same as original)."""
        styles = {
            "ready": ("Start Recording", "#1E5631", "#2E8B57"),
            "recording": ("Stop Recording", "#8B0000", "#A52A2A"),
            "processing": ("Processing...", "#8B4500", "#CD6600")
        }
        text, bg_color, hover_color = styles.get(state, styles["ready"])
        self.recording_button.setText(text)
        self.recording_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)

    def toggle_recording(self):
        """Toggle recording with safety checks."""
        if self.state.has_active_threads:
            logger.warning("Cannot start recording: processing still in progress")
            QMessageBox.information(self, "Processing",
                                    "Please wait for the current task to complete.")
            return

        if not self.state.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording with confidence display reset."""
        self.set_button_style("recording")
        self.progress_bar.setValue(0)
        self.state.is_recording = True

        # Reset confidence and quality displays
        self.confidence_label.setText("--")
        self.confidence_label.setToolTip("")  # clear any previous tooltip
        self.confidence_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 60px;
            }
        """)
        self.quality_label.setText("--")
        self.quality_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 80px;
            }
        """)

        threading.Thread(target=self.record_audio_background, daemon=True).start()

    def record_audio_background(self):
        """Background recording with proper signal usage."""
        try:
            with AudioRecorder(self.state) as recorder:
                audio_data = recorder.record()
        except Exception as e:
            logger.error(f"Recording failed: {e}", exc_info=True)
            audio_data = None

        self.state.is_recording = False

        if audio_data:
            self._transcription_ready.emit(audio_data)
        else:
            self._recording_failed.emit()

    def stop_recording(self):
        """Stop recording."""
        self.state.is_recording = False
        self.set_button_style("processing")

    def start_transcription(self, audio_data):
        """Start transcription with safe thread management - no terminate()."""
        if self.state.has_active_threads:
            logger.warning("Cannot start transcription: previous task still in progress")
            QMessageBox.information(self, "Processing",
                                    "Previous audio is still being processed. Please wait.")
            return

        if self.current_worker:
            self._disconnect_worker_signals()

            if self.current_worker.isRunning():
                if hasattr(self.current_worker, 'cancel'):
                    self.current_worker.cancel()

                progress_dialog = QProgressDialog("Finishing previous task...", None, 0, 0, self)
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.show()

                wait_time = 0
                max_wait = 10000

                while self.current_worker.isRunning() and wait_time < max_wait:
                    QApplication.processEvents()
                    self.current_worker.wait(100)
                    wait_time += 100

                progress_dialog.close()

                if self.current_worker.isRunning():
                    QMessageBox.warning(self, "Processing Delayed",
                                        "Previous task is taking longer than expected. "
                                        "Please try again in a few moments.")
                    return

            self.current_worker.deleteLater()
            self.current_worker = None

        self.current_worker = TranscriptionThread(
            audio_data,
            self.model_selector.currentText(),
            self.state
        )

        self._connect_worker_signals(self.current_worker)

        self.current_worker.start()
        logger.info("Started new transcription worker")

    def re_refine_text(self):
        """Re-refine with safe thread management - no terminate()."""
        text = self.transcription_box.toPlainText().strip()
        if not text:
            return

        if self.state.has_active_threads:
            logger.warning("Cannot start re-refinement: processing still in progress")
            QMessageBox.information(self, "Processing",
                                    "Please wait for the current task to complete.")
            return

        if self.current_worker and self.current_worker.isRunning():
            logger.info("Waiting for previous worker...")

            if hasattr(self.current_worker, 'cancel'):
                self.current_worker.cancel()

            self.current_worker.wait(5000)

            if self.current_worker.isRunning():
                QMessageBox.warning(self, "Processing",
                                    "Previous task still running. Please try again.")
                return

            self._disconnect_worker_signals()

        self.set_button_style("processing")
        self.progress_bar.setValue(50)
        self.current_worker = RefinementThread(text, self.model_selector.currentText(), self.state)

        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self.handle_error)
        self._worker_connections.append((self.current_worker.refinement_finished, self.display_refined_text))
        self._worker_connections.append((self.current_worker.error_occurred, self.handle_error))

        self.current_worker.start()

    def display_transcription(self, text):
        """Display transcription with confidence update."""
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

        audio_analysis = self.state.get_audio_analysis()
        if audio_analysis:
            self.update_quality_display(audio_analysis.quality)

    def display_refined_text(self, text):
        """Display refined text."""
        self.refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style("ready")

    def copy_text(self, widget):
        """Copy text."""
        clipboard = QApplication.clipboard()
        clipboard.setText(widget.toPlainText())

    def update_confidence_display(self, confidence: float):
        """Update the confidence display with color coding and 'N/A' fallback."""
        # Handle NaN or sentinel values gracefully
        if confidence is None or math.isnan(confidence) or confidence < 0:
            self.confidence_label.setText("N/A")
            self.confidence_label.setToolTip("Confidence not calculated (word timestamps disabled).")
            self.confidence_label.setStyleSheet("""
                QLabel {
                    background-color: #2c3e50;
                    color: white;
                    border: 1px solid #bdc3c7;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    min-width: 60px;
                }
            """)
            return

        percentage = confidence * 100.0
        self.confidence_label.setText(f"{percentage:.1f}%")
        self.confidence_label.setToolTip("Segment-level proxy confidence")

        if confidence >= 0.8:
            color = "#27ae60"  # Green
        elif confidence >= 0.6:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Red

        self.confidence_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 60px;
            }}
        """)

    def update_quality_display(self, quality: AudioQuality):
        """Update the audio quality display with color coding."""
        quality_text = quality.value.capitalize()
        self.quality_label.setText(quality_text)

        quality_colors = {
            AudioQuality.EXCELLENT: "#27ae60",
            AudioQuality.GOOD: "#2ecc71",
            AudioQuality.FAIR: "#f39c12",
            AudioQuality.POOR: "#e74c3c"
        }

        color = quality_colors.get(quality, "#95a5a6")

        self.quality_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 80px;
            }}
        """)

    def handle_error(self, error_message):
        """Handle errors."""
        logger.error(error_message)
        QMessageBox.critical(self, "Error", error_message)
        self.set_button_style("ready")
        self.progress_bar.setValue(0)

    def on_sensitivity_changed(self, text):
        """SENSITIVITY: Handle sensitivity level changes."""
        level_map = {
            "Original": SensitivityLevel.ORIGINAL,
            "Balanced": SensitivityLevel.BALANCED,
            "Sensitive": SensitivityLevel.SENSITIVE
        }

        new_level = level_map.get(text, SensitivityLevel.BALANCED)
        GLOBAL_AUDIO_CONFIG.current_level = new_level

        config = GLOBAL_AUDIO_CONFIG.config
        info_text = f"Sensitivity: {text}\n\n"

        if new_level == SensitivityLevel.SENSITIVE:
            info_text += "• Maximum amplification enabled (4x)\n"
            info_text += "• Aggressive noise reduction active\n"
            info_text += "• Very lenient quality thresholds\n"
            info_text += "• Optimized for quiet speakers"
        elif new_level == SensitivityLevel.BALANCED:
            info_text += "• Balanced amplification (2.5x)\n"
            info_text += "• Moderate noise reduction\n"
            info_text += "• Improved quality thresholds\n"
            info_text += "• Good for most users"
        else:
            info_text += "• Conservative settings\n"
            info_text += "• Minimal processing\n"
            info_text += "• Original thresholds\n"
            info_text += "• Best for clear recordings"

        QMessageBox.information(self, "Sensitivity Changed", info_text)

    def closeEvent(self, event):
        """Handle close event with proper cleanup."""
        if self.state.has_active_threads:
            logger.info("Waiting for active threads to finish...")

            if self.current_worker and hasattr(self.current_worker, 'cancel'):
                self.current_worker.cancel()

            wait_time = 0
            while self.state.has_active_threads and wait_time < 3000:
                QApplication.processEvents()
                wait_time += 100

        if AudioRecorder._pyaudio_instance:
            try:
                AudioRecorder._pyaudio_instance.terminate()
                logger.info("PyAudio singleton terminated")
            except:
                pass

        event.accept()


# --------------------------
# Application Entry Point
# --------------------------
def main() -> None:
    """Main entry point for the Enhanced Hybrid Audio Transcriber."""
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
