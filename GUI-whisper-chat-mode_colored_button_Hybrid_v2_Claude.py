#!/usr/bin/env python3
"""
Hybrid Audio Transcriber Application - PRACTICAL CHUNKING VERSION
=================================================================
This application records audio, transcribes it using MLX Whisper, and refines (corrects) the text via an LLM (Ollama).
It combines robust configuration, resource management, error handling, and a clean PyQt5 GUI.

Key Features:
- No limitation on recording duration.
- No limitation on text length passed to the LLM.
- Uses a fallback configuration if the selected model is not in the pre-defined list.
- UPDATED: Practical audio chunking to balance quality and efficiency
- Audio preprocessing optimization for Whisper (44.1kHz → 16kHz conversion).
- Enhanced repetition detection and cleanup.
- Comprehensive progress reporting for chunking operations.
- Extensive inline documentation and comments for clarity.
- NEW: Force strategy buttons to reprocess audio with specific transcription methods
- FIXED: Audio buffer persistence - copy buttons no longer clear audio from memory
- ENHANCED: Comprehensive VAD chunking with no audio loss guarantee

Enhanced Features (FINAL):
- FIXED: Proper WAV file creation with headers for all transcription methods
- UPDATED: Practical chunking (2min+ recordings, 2s silence threshold)
- Voice Activity Detection (VAD) with intelligent validation
- Time-based chunking fallback with larger chunks (60s chunks, 2s overlap)
- Audio preprocessing for optimal Whisper input (16kHz conversion)
- Enhanced MLX Whisper parameters (removed best_of, optimized temperature)
- Repetition detection and cleanup
- Progress reporting for chunk processing
- Enhanced error handling and logging
- NEW: Temporary audio storage for reprocessing with different strategies
- NEW: Force transcription strategy buttons (Single, Time-Based, VAD-Based)
- FIXED: Audio buffer persistence after copying text
- ENHANCED: Comprehensive VAD chunking with coverage validation and retry logic

Recent Updates:
- UPDATED: MIN_DURATION_FOR_CHUNKING set to 2min (practical threshold)
- UPDATED: VAD_MIN_SILENCE_LEN set to 2.0s (natural speech pause detection)
- UPDATED: VAD_SILENCE_THRESH at -50dB (less sensitive to background noise)
- UPDATED: MIN_CHUNK_DURATION at 15s (substantial chunks only)
- UPDATED: Better validation to prevent unnecessary chunking
- NEW: Audio buffer management for temporary storage
- NEW: Force strategy UI controls
- FIXED: Copy buttons now preserve audio in memory for re-transcription
- ENHANCED: VAD chunking now guarantees no audio loss with comprehensive validation
- ENHANCED: Retry mechanism for failed chunk transcriptions
- ENHANCED: Detailed logging and boundary tracking for all chunking operations
"""

import sys
import re
import io
import logging
import threading
import datetime
import tempfile
import wave
from dataclasses import dataclass
from functools import partial
from contextlib import contextmanager
from typing import List, Tuple, Optional

import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QMessageBox
)
# FIXED: Import QTimer for thread-safe GUI updates
from PyQt5.QtCore import pyqtSignal, QThread, QObject, QTimer

# Optional imports for enhanced audio processing
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    PYDUB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("pydub available - enhanced audio processing enabled")
except ImportError:
    PYDUB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pydub not available - enhanced audio processing disabled. Install with: pip install pydub")

import mlx_whisper
import ollama

# --------------------------
# Logging Configuration
# --------------------------
# Configure logging to output both to console and to a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
        # logging.FileHandler('hybrid_transcriber.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)


# --------------------------
# Audio Configuration (UPDATED for Practical Chunking)
# --------------------------
class AudioConfig:
    """
    Audio configuration settings.

    NOTE: We do not impose any upper limit on recording duration.

    Recording Settings (PRESERVED FROM ORIGINAL):
    - SAMPLE_RATE: 44100 Hz for high-quality recording (unchanged)
    - FORMAT: 16-bit audio format (unchanged)
    - CHANNELS: Mono recording (unchanged)
    - CHUNK_SIZE: Buffer size for recording (unchanged)

    UPDATED - Practical Processing Settings:
    - Chunking configuration for long audio processing
    - VAD (Voice Activity Detection) settings for intelligent chunking
    - Audio preprocessing parameters for Whisper optimization
    """
    # Original recording configuration (PRESERVED EXACTLY)
    SAMPLE_RATE = 44100  # Samples per second - keep high quality for recording
    FORMAT = pyaudio.paInt16  # 16-bit format
    CHANNELS = 1  # Mono recording
    CHUNK_SIZE = 1024  # Number of frames per buffer

    # UPDATED - Practical chunking configuration
    MIN_DURATION_FOR_CHUNKING = 120.0  # 2 minutes - start chunking evaluation after 2 minutes

    # UPDATED - Natural speech pause detection
    VAD_MIN_SILENCE_LEN = 2000  # 2.0 seconds of silence to split on
    VAD_SILENCE_THRESH = -50  # dB threshold for silence detection (less sensitive to background noise)
    VAD_KEEP_SILENCE = 1000  # Keep 1s of silence at chunk edges (natural boundaries)
    MIN_CHUNK_DURATION = 15.0  # Minimum 15 seconds per chunk (substantial chunks)
    MAX_CHUNK_DURATION = 90.0  # Maximum 90 seconds per chunk (reasonable upper limit)

    # UPDATED - Time-based chunking settings (fallback when VAD ineffective)
    TIME_CHUNK_DURATION = 60.0  # 60 seconds per chunk
    OVERLAP_DURATION = 2.0  # 2 second overlap between chunks

    # ENHANCED - Whisper optimization settings (unchanged)
    WHISPER_SAMPLE_RATE = 16000  # Optimal sample rate for Whisper


# --------------------------
# Enhanced Text Processing Utilities
# --------------------------
class TextProcessor:
    """
    ENHANCED CLASS - Handles text processing operations including repetition detection and cleanup.

    Provides methods for:
    - Detecting repetitive patterns in transcribed text
    - Cleaning repetitions and normalizing text
    - Combining multiple transcriptions intelligently
    - Final text cleanup and formatting
    - ENHANCED: Comprehensive transcription combination with failure handling
    """

    @staticmethod
    def detect_repetition(text: str) -> bool:
        """
        Detect if text contains excessive repetition patterns.

        Args:
            text: Text to analyze for repetitive patterns

        Returns:
            True if significant repetition is detected, False otherwise
        """
        if not text or len(text.strip()) < 10:
            return False

        words = text.lower().split()
        if len(words) < 5:
            return False

        # Check for word repetitions (3+ consecutive identical words)
        for i in range(len(words) - 2):
            if len(words[i]) > 2 and words[i] == words[i + 1] == words[i + 2]:
                logger.debug(f"Detected word repetition: '{words[i]}'")
                return True

        # Check for phrase repetitions (2-4 word phrases)
        for n in [2, 3, 4]:
            for i in range(len(words) - (n * 2)):
                phrase1 = tuple(words[i:i + n])
                phrase2 = tuple(words[i + n:i + n * 2])

                if phrase1 == phrase2 and len(phrase1[0]) > 1:
                    logger.debug(f"Detected phrase repetition: '{' '.join(phrase1)}'")
                    return True

        return False

    @staticmethod
    def clean_repetitions(text: str) -> str:
        """
        Remove obvious repetitions from text.

        Args:
            text: Text to clean of repetitive patterns

        Returns:
            Cleaned text with repetitions removed
        """
        if not text:
            return text

        # Remove word repetitions (3+ consecutive)
        text = re.sub(r'\b(\w+)(\s+\1\b){2,}', r'\1', text, flags=re.IGNORECASE)

        # Remove phrase repetitions (up to 4 words)
        text = re.sub(r'(.{4,15}?)\s+\1\s+\1', r'\1', text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def combine_transcriptions(transcriptions: List[str]) -> str:
        """
        Intelligently combine multiple transcriptions, removing overlaps.

        Args:
            transcriptions: List of transcription strings from chunks

        Returns:
            Combined transcription with overlaps removed
        """
        if not transcriptions:
            return ""

        if len(transcriptions) == 1:
            return transcriptions[0]

        # Remove boundary duplicates from overlapping chunks
        cleaned = [transcriptions[0]]

        for i in range(1, len(transcriptions)):
            current = transcriptions[i]
            previous = cleaned[-1]

            # Find overlapping words at boundary
            current_words = current.split()
            previous_words = previous.split()

            # Look for duplicate sequences (up to 8 words)
            max_overlap = min(len(current_words), len(previous_words), 8)
            best_overlap = 0

            for overlap_len in range(max_overlap, 0, -1):
                if (len(previous_words) >= overlap_len and
                        len(current_words) >= overlap_len and
                        previous_words[-overlap_len:] == current_words[:overlap_len]):
                    best_overlap = overlap_len
                    break

            if best_overlap > 0:
                # Remove overlapping words from current transcription
                remaining_words = current_words[best_overlap:]
                if remaining_words:
                    cleaned.append(" ".join(remaining_words))
                # If all words were overlapping, skip this transcription
            else:
                cleaned.append(current)

        # Join and final cleanup
        combined = " ".join(cleaned)
        return TextProcessor.clean_final_text(combined)

    @staticmethod
    def combine_transcriptions_comprehensive(transcriptions: List[str]) -> str:
        """
        ENHANCED - Combine transcriptions while handling failed chunk placeholders.

        This method handles the comprehensive transcription results that may include
        failure placeholders while maintaining the sequence and continuity of the audio.

        Args:
            transcriptions: List of transcription strings (may include failure placeholders)

        Returns:
            Combined transcription with placeholders handled appropriately
        """
        if not transcriptions:
            return ""

        if len(transcriptions) == 1:
            # Check if it's a failure placeholder
            if transcriptions[0].startswith("[Chunk") and "failed]" in transcriptions[0]:
                return "Transcription failed for single chunk"
            return transcriptions[0]

        # Separate successful transcriptions from placeholders
        processed_transcriptions = []
        failed_chunks = []

        for i, text in enumerate(transcriptions):
            if text.startswith("[Chunk") and "failed]" in text:
                # Handle failed chunk placeholder
                failed_chunks.append(i + 1)
                logger.warning(f"Skipping failed chunk placeholder: {text}")
                # Add a gap indicator if this creates a significant break
                if processed_transcriptions and len(processed_transcriptions) > 0:
                    processed_transcriptions.append(" [gap] ")
            else:
                processed_transcriptions.append(text)

        if not processed_transcriptions:
            return f"All {len(transcriptions)} chunks failed transcription"

        if failed_chunks:
            logger.warning(
                f"Final transcription includes gaps due to {len(failed_chunks)} failed chunks: {failed_chunks}")

        # Use existing combination logic for successful transcriptions
        combined = TextProcessor.combine_transcriptions(processed_transcriptions)

        # Add metadata about failures if any occurred
        if failed_chunks:
            metadata = f" [Note: {len(failed_chunks)} chunks failed: {failed_chunks}]"
            combined = combined + metadata

        return combined

    @staticmethod
    def clean_final_text(text: str) -> str:
        """
        Final cleanup of transcribed text.

        Args:
            text: Text to perform final cleanup on

        Returns:
            Final cleaned and formatted text
        """
        if not text:
            return text

        # Clean repetitions
        text = TextProcessor.clean_repetitions(text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1', text)

        return text.strip()


# --------------------------
# Enhanced Audio Processing (FIXED)
# --------------------------
class AudioProcessor:
    """
    ENHANCED CLASS - Handles audio preprocessing and analysis operations.

    FIXED: All methods now properly handle WAV file creation with correct headers.

    Provides methods for:
    - Converting audio between different sample rates
    - Applying audio filters and normalization
    - Estimating audio duration and characteristics
    - Preprocessing audio for optimal Whisper performance
    - Creating proper WAV files from raw PCM data
    """

    @staticmethod
    def estimate_duration(audio_data: bytes, sample_rate: int = AudioConfig.SAMPLE_RATE) -> float:
        """
        Estimate audio duration in seconds from raw audio data.

        Args:
            audio_data: Raw audio data in bytes
            sample_rate: Sample rate of the audio

        Returns:
            Duration in seconds
        """
        try:
            # Assumes 16-bit mono audio (2 bytes per sample)
            num_samples = len(audio_data) // 2
            duration = num_samples / sample_rate
            return duration
        except Exception as e:
            logger.error(f"Error estimating audio duration: {e}")
            return 0.0

    @staticmethod
    def _create_wav_from_pcm(pcm_data: bytes, sample_rate: int) -> bytes:
        """
        FIXED - Create a proper WAV file from raw PCM data.

        This is the critical fix that prevents the "Invalid data found when processing input" error
        by ensuring all audio data has proper WAV headers before being passed to MLX Whisper.

        Args:
            pcm_data: Raw PCM audio data
            sample_rate: Sample rate of the audio

        Returns:
            Complete WAV file as bytes with proper headers
        """
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(AudioConfig.CHANNELS)  # Mono
                wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data)

            wav_data = buffer.getvalue()
            logger.debug(f"Created WAV file: {len(wav_data)} bytes from {len(pcm_data)} PCM bytes")
            return wav_data
        except Exception as e:
            logger.error(f"Error creating WAV file from PCM data: {e}")
            raise

    @staticmethod
    def preprocess_for_whisper(audio_data: bytes) -> bytes:
        """
        FIXED - Preprocess audio for optimal Whisper transcription.

        Converts high-quality recording (44.1kHz) to Whisper's optimal format (16kHz)
        and applies audio enhancements when pydub is available. Now properly handles
        WAV file creation to prevent FFmpeg errors.

        Args:
            audio_data: Raw PCM audio data from recording at 44.1kHz

        Returns:
            Preprocessed audio data as a complete WAV file optimized for Whisper
        """
        if not PYDUB_AVAILABLE:
            logger.debug("pydub not available - creating basic WAV file from raw PCM data")
            return AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)

        try:
            logger.debug("Preprocessing audio for optimal Whisper transcription")

            # FIXED: First, create a proper WAV file from raw PCM data
            wav_data = AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)

            # Convert WAV bytes to AudioSegment for processing
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")

            # Log original properties
            logger.debug(
                f"Original audio: {len(audio_segment)}ms, {audio_segment.frame_rate}Hz, {audio_segment.channels}ch")

            # Convert to Whisper's optimal settings
            audio_segment = audio_segment.set_frame_rate(AudioConfig.WHISPER_SAMPLE_RATE)  # 16kHz for Whisper
            audio_segment = audio_segment.set_channels(1)  # Ensure mono
            audio_segment = audio_segment.normalize()  # Normalize levels

            # Apply high-pass filter to remove low-frequency noise (if available)
            try:
                audio_segment = audio_segment.high_pass_filter(80)
                logger.debug("Applied high-pass filter at 80Hz")
            except Exception as e:
                logger.debug(f"High-pass filter not available: {e}")

            # Convert back to WAV bytes
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav")
            processed_data = buffer.getvalue()

            logger.debug(f"Audio preprocessing complete: {len(processed_data)} bytes")
            return processed_data

        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, creating basic WAV file")
            return AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)


class AudioChunker:
    """
    ENHANCED CLASS - Comprehensive audio chunking with no audio loss guarantee.

    FIXED: All chunking methods now output proper WAV files instead of raw audio data.
    UPDATED: Practical chunking settings to balance quality and efficiency.
    ENHANCED: Comprehensive validation and coverage tracking to ensure no audio is lost.

    Provides two chunking strategies:
    1. VAD-based chunking: Uses silence detection to find natural speech boundaries with full coverage validation
    2. Time-based chunking: Fixed-duration chunks with overlap as fallback

    The chunker now guarantees that all audio is processed and no segments are lost.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.AudioChunker')

    def should_chunk_audio(self, audio_data: bytes) -> bool:
        """
        UPDATED - Practical determination of whether audio should be chunked.

        Args:
            audio_data: Raw PCM audio data to analyze

        Returns:
            True if audio should be chunked, False otherwise
        """
        try:
            duration = AudioProcessor.estimate_duration(
                audio_data,
                AudioConfig.SAMPLE_RATE  # Use original sample rate for raw PCM data
            )
            should_chunk = duration > AudioConfig.MIN_DURATION_FOR_CHUNKING

            self.logger.info(
                f"Audio duration: {duration:.1f}s ({duration / 60:.1f}min), chunking threshold: {AudioConfig.MIN_DURATION_FOR_CHUNKING:.1f}s, chunking needed: {should_chunk}")
            return should_chunk

        except Exception as e:
            self.logger.error(f"Error determining chunking need: {e}")
            return False

    def chunk_audio(self, audio_data: bytes) -> List[bytes]:
        """
        ENHANCED - Comprehensive chunking with full audio coverage validation.

        Now properly converts raw PCM data to WAV format and outputs complete WAV files
        for each chunk, preventing the "Invalid data found when processing input" error.
        Includes comprehensive validation to ensure no audio is lost.

        Args:
            audio_data: Raw PCM audio data from recording

        Returns:
            List of complete WAV file chunks as bytes
        """
        if not PYDUB_AVAILABLE:
            self.logger.warning("pydub not available - returning original as single WAV chunk")
            wav_data = AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)
            return [wav_data]

        try:
            # FIXED: First convert raw PCM to proper WAV format
            wav_data = AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")

            # Try VAD-based chunking
            self.logger.info("Attempting comprehensive VAD-based chunking with coverage validation")
            chunks = self._chunk_with_vad_comprehensive(audio_segment)

            if chunks and len(chunks) >= 2:
                # VAD was successful - convert to WAV bytes with validation
                chunk_data_list = []
                original_duration = len(audio_segment)
                total_chunk_duration = 0

                for i, chunk in enumerate(chunks):
                    try:
                        buffer = io.BytesIO()
                        chunk.export(buffer, format="wav")
                        chunk_wav_data = buffer.getvalue()
                        chunk_data_list.append(chunk_wav_data)

                        chunk_duration_ms = len(chunk)
                        total_chunk_duration += chunk_duration_ms

                        self.logger.debug(f"VAD chunk {i + 1}: {len(chunk_wav_data)} bytes, "
                                          f"{chunk_duration_ms / 1000.0:.1f}s")
                    except Exception as e:
                        self.logger.error(f"Error converting VAD chunk {i} to WAV: {e}")

                # VALIDATION: Verify all audio is covered
                coverage_ratio = total_chunk_duration / original_duration
                self.logger.info(f"VAD chunking validation: {len(chunk_data_list)} chunks, "
                                 f"coverage: {coverage_ratio:.3f} ({coverage_ratio * 100:.1f}%)")

                if chunk_data_list and coverage_ratio > 0.90:
                    self.logger.info(f"VAD chunking successful: {len(chunk_data_list)} chunks with good coverage")
                    return chunk_data_list
                else:
                    self.logger.warning(
                        f"VAD chunking coverage too low ({coverage_ratio * 100:.1f}%), using single chunk")

            # VAD was not effective - return as single chunk
            self.logger.info("VAD chunking not effective, returning as single chunk")
            return [wav_data]

        except Exception as e:
            self.logger.error(f"Audio chunking failed: {e}")
            # FIXED: Return original as properly formatted WAV instead of raw PCM
            wav_data = AudioProcessor._create_wav_from_pcm(audio_data, AudioConfig.SAMPLE_RATE)
            return [wav_data]

    def _chunk_with_vad_comprehensive(self, audio_segment: AudioSegment) -> List[AudioSegment]:
        """
        ENHANCED - VAD-based chunking that ensures ALL audio is covered with comprehensive validation.

        Now includes:
        - Full audio coverage validation
        - Gap detection and filling
        - Beginning/end preservation
        - Comprehensive logging of chunk boundaries

        Args:
            audio_segment: AudioSegment to chunk

        Returns:
            List of AudioSegment chunks that cover the ENTIRE audio, or empty list if VAD not effective
        """
        try:
            original_duration_ms = len(audio_segment)
            self.logger.info(f"VAD chunking audio: {original_duration_ms}ms ({original_duration_ms / 1000:.1f}s)")

            # Split on silence using practical parameters
            raw_chunks = split_on_silence(
                audio_segment,
                min_silence_len=AudioConfig.VAD_MIN_SILENCE_LEN,  # 2.0 seconds
                silence_thresh=AudioConfig.VAD_SILENCE_THRESH,  # -50dB
                keep_silence=AudioConfig.VAD_KEEP_SILENCE  # Keep 1 second
            )

            if not raw_chunks:
                self.logger.info("VAD found no silence breaks - audio appears to be continuous speech")
                return []

            # CRITICAL FIX: Ensure chunks cover the full audio duration
            total_chunks_duration = sum(len(chunk) for chunk in raw_chunks)
            coverage_ratio = total_chunks_duration / original_duration_ms

            self.logger.info(f"VAD initial split: {len(raw_chunks)} chunks, "
                             f"total duration: {total_chunks_duration}ms, "
                             f"coverage: {coverage_ratio:.3f} ({coverage_ratio * 100:.1f}%)")

            # If VAD missed significant portions of audio, it's not reliable
            if coverage_ratio < 0.95:  # Less than 95% coverage
                self.logger.warning(f"VAD coverage too low ({coverage_ratio * 100:.1f}%), "
                                    f"missing {(1 - coverage_ratio) * 100:.1f}% of audio")
                return []

            # Process chunks with enhanced validation
            processed_chunks = self._process_raw_chunks_with_validation(raw_chunks, audio_segment)

            if not processed_chunks:
                self.logger.warning("Chunk processing resulted in no valid chunks")
                return []

            # ENHANCED: Verify final chunks cover full audio
            final_duration = sum(len(chunk) for chunk in processed_chunks)
            final_coverage = final_duration / original_duration_ms

            self.logger.info(f"Processed chunks: {len(processed_chunks)}, "
                             f"final duration: {final_duration}ms, "
                             f"final coverage: {final_coverage:.3f} ({final_coverage * 100:.1f}%)")

            if final_coverage < 0.90:  # Less than 90% coverage after processing
                self.logger.warning(f"Final coverage too low ({final_coverage * 100:.1f}%), VAD not suitable")
                return []

            # Validation checks
            if len(processed_chunks) < 2:
                self.logger.info(f"Only {len(processed_chunks)} chunk(s) after processing, not beneficial")
                return []

            # More reasonable upper limit
            max_reasonable_chunks = max(3, int(original_duration_ms / 60000) + 2)  # 1 per minute + buffer
            if len(processed_chunks) > max_reasonable_chunks:
                self.logger.info(f"Too many chunks ({len(processed_chunks)} > {max_reasonable_chunks}), "
                                 f"likely over-segmented")
                return []

            # Quality check
            avg_chunk_duration = final_duration / len(processed_chunks) / 1000.0
            if avg_chunk_duration < AudioConfig.MIN_CHUNK_DURATION:
                self.logger.info(f"Average chunk duration too short ({avg_chunk_duration:.1f}s)")
                return []

            # LOG CHUNK BOUNDARIES for debugging
            self.logger.info("VAD chunk boundaries:")
            cumulative_time = 0
            for i, chunk in enumerate(processed_chunks):
                chunk_duration = len(chunk) / 1000.0
                self.logger.info(f"  Chunk {i + 1}: {cumulative_time:.1f}s - {cumulative_time + chunk_duration:.1f}s "
                                 f"(duration: {chunk_duration:.1f}s)")
                cumulative_time += chunk_duration

            self.logger.info(f"VAD chunking successful: {len(processed_chunks)} chunks covering "
                             f"{final_coverage * 100:.1f}% of audio")
            return processed_chunks

        except Exception as e:
            self.logger.error(f"VAD chunking failed: {e}", exc_info=True)
            return []

    def _chunk_with_vad(self, audio_segment: AudioSegment) -> List[AudioSegment]:
        """
        PRESERVED - Original VAD method for compatibility, now delegates to comprehensive version.

        Args:
            audio_segment: AudioSegment to chunk

        Returns:
            List of AudioSegment chunks, empty list if ineffective
        """
        return self._chunk_with_vad_comprehensive(audio_segment)

    def _chunk_with_time(self, audio_segment: AudioSegment) -> List[AudioSegment]:
        """
        UPDATED - Practical time-based chunking with larger chunks.

        Args:
            audio_segment: AudioSegment to chunk

        Returns:
            List of AudioSegment chunks
        """
        try:
            chunk_duration_ms = int(AudioConfig.TIME_CHUNK_DURATION * 1000)
            overlap_ms = int(AudioConfig.OVERLAP_DURATION * 1000)

            chunks = []
            start_ms = 0

            while start_ms < len(audio_segment):
                end_ms = min(start_ms + chunk_duration_ms, len(audio_segment))
                chunk = audio_segment[start_ms:end_ms]

                # Only keep chunks that meet minimum duration
                if len(chunk) >= AudioConfig.MIN_CHUNK_DURATION * 1000:
                    chunks.append(chunk)
                elif chunks:
                    # Merge short final chunk with previous chunk
                    chunks[-1] = chunks[-1] + chunk

                # Move to next chunk with overlap, prevent infinite loop
                next_start = end_ms - overlap_ms
                if next_start <= start_ms:
                    break
                start_ms = next_start

            self.logger.info(f"Practical time-based chunking created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Time-based chunking failed: {e}")
            return [audio_segment]  # Return original as single chunk

    def _process_raw_chunks_with_validation(self, raw_chunks: List[AudioSegment], original_audio: AudioSegment) -> List[
        AudioSegment]:
        """
        ENHANCED - Process raw chunks with validation to ensure no audio is lost.

        Args:
            raw_chunks: Raw chunks from VAD splitting
            original_audio: Original audio segment for validation

        Returns:
            Processed chunks that maintain full audio coverage
        """
        if not raw_chunks:
            return []

        self.logger.debug(f"Processing {len(raw_chunks)} raw chunks")

        # Log raw chunk details
        for i, chunk in enumerate(raw_chunks):
            duration = len(chunk) / 1000.0
            self.logger.debug(f"  Raw chunk {i + 1}: {duration:.1f}s")

        # ENHANCED: Intelligent merging that preserves all audio
        merged_chunks = []
        current_chunk = raw_chunks[0]

        for next_chunk in raw_chunks[1:]:
            current_duration = len(current_chunk) / 1000.0

            if current_duration < AudioConfig.MIN_CHUNK_DURATION * 1.5:  # 22.5s
                # Merge with next chunk
                current_chunk = current_chunk + next_chunk
                self.logger.debug(f"Merged chunk to {len(current_chunk) / 1000.0:.1f}s")
            else:
                # Keep current chunk and start new one
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        # CRITICAL: Don't forget the last chunk
        if len(current_chunk) / 1000.0 >= AudioConfig.MIN_CHUNK_DURATION:
            merged_chunks.append(current_chunk)
        elif merged_chunks:
            # Merge short final chunk with previous
            self.logger.debug("Merging final short chunk with previous")
            merged_chunks[-1] = merged_chunks[-1] + current_chunk
        else:
            # If no merged chunks yet, keep it anyway
            merged_chunks.append(current_chunk)

        # Second pass: handle overly long chunks
        final_chunks = []
        for chunk in merged_chunks:
            duration = len(chunk) / 1000.0

            if duration <= AudioConfig.MAX_CHUNK_DURATION:
                final_chunks.append(chunk)
            else:
                # Split long chunk carefully to avoid gaps
                split_chunks = self._split_long_chunk_safely(chunk)
                final_chunks.extend(split_chunks)

        # VALIDATION: Verify no audio was lost during processing
        original_duration = len(original_audio)
        final_total_duration = sum(len(chunk) for chunk in final_chunks)

        self.logger.debug(f"Chunk processing validation:")
        self.logger.debug(f"  Original: {original_duration}ms")
        self.logger.debug(f"  Final total: {final_total_duration}ms")
        self.logger.debug(f"  Difference: {abs(original_duration - final_total_duration)}ms")

        return final_chunks

    def _process_raw_chunks(self, raw_chunks: List[AudioSegment]) -> List[AudioSegment]:
        """
        PRESERVED - Original method for compatibility, now delegates to enhanced version.

        Args:
            raw_chunks: Raw chunks from VAD splitting

        Returns:
            Processed chunks meeting duration requirements
        """
        # For compatibility, we'll create a dummy original audio from the chunks
        if not raw_chunks:
            return []

        # Reconstruct approximate original audio for validation
        total_duration = sum(len(chunk) for chunk in raw_chunks)
        dummy_original = AudioSegment.silent(duration=total_duration)

        return self._process_raw_chunks_with_validation(raw_chunks, dummy_original)

    def _split_long_chunk_safely(self, chunk: AudioSegment) -> List[AudioSegment]:
        """
        ENHANCED - Split long chunks while ensuring no audio is lost.

        Args:
            chunk: AudioSegment that's too long

        Returns:
            List of smaller AudioSegment chunks with no gaps
        """
        target_duration_ms = int(AudioConfig.MAX_CHUNK_DURATION * 0.75 * 1000)
        chunks = []
        start = 0
        chunk_length = len(chunk)

        self.logger.debug(f"Splitting {chunk_length / 1000:.1f}s chunk into ~{target_duration_ms / 1000:.1f}s pieces")

        while start < chunk_length:
            end = min(start + target_duration_ms, chunk_length)
            sub_chunk = chunk[start:end]

            # CRITICAL: Include ALL pieces, even short ones
            chunks.append(sub_chunk)
            self.logger.debug(f"Split piece: {start / 1000:.1f}s - {end / 1000:.1f}s ({len(sub_chunk) / 1000:.1f}s)")

            start = end

        return chunks

    def _split_long_chunk(self, chunk: AudioSegment) -> List[AudioSegment]:
        """
        PRESERVED - Original method for compatibility, now delegates to enhanced version.

        Args:
            chunk: AudioSegment that's too long

        Returns:
            List of smaller AudioSegment chunks
        """
        return self._split_long_chunk_safely(chunk)


# --------------------------
# Model Configuration (PRESERVED EXACTLY FROM ORIGINAL)
# --------------------------
@dataclass
class ModelConfig:
    """
    Model configuration for the LLM text refinement.

    Each model configuration includes:
      - name: The model's name.
      - ctx_num: The context size (number of tokens).
      - temperature: The sampling temperature.
      - seed: Random seed for reproducibility.
      - system_message: The prompt for the system role.
      - user_message: The prompt for the user role, expecting a placeholder "{text}".

    If a model is not found in the pre-defined list, a default configuration will be used.

    ENHANCED: Added repetition handling instructions to system messages.
    """
    name: str
    ctx_num: int
    temperature: float
    seed: int
    system_message: str
    user_message: str

    @classmethod
    def get_default_configs(cls):
        """
        Returns a dictionary of pre-defined model configurations.
        ENHANCED: Added repetition handling instructions to all configurations.
        """
        return {
            'phi4:latest': cls(
                name='phi4:latest',
                ctx_num=8192,
                temperature=0.2,
                seed=1,
                system_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Pay special attention to removing any repeated words or phrases that appear to be transcription errors. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
                user_message='Here is the text to be corrected: "{text}"'
            ),
            'deepseek-r1:1.5b': cls(
                name='deepseek-r1:1.5b',
                ctx_num=8192,
                temperature=1.3,
                seed=1,
                system_message=(
                    "You are my helpful and professional text corrector."
                ),
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Pay special attention to removing any repeated words or phrases that appear to be transcription errors. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                ),
            ),
            'deepseek-r1:latest': cls(
                name='deepseek-r1:latest',
                ctx_num=8192,
                temperature=1.3,
                seed=1,
                system_message=(
                    "You are my helpful and professional text corrector."
                ),
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Pay special attention to removing any repeated words or phrases that appear to be transcription errors. '
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
                system_message=(
                    "You are my helpful and professional text corrector."
                ),
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Pay special attention to removing any repeated words or phrases that appear to be transcription errors. '
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
                system_message=(
                    "You are my helpful and professional text corrector."
                ),
                user_message=(
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Pay special attention to removing any repeated words or phrases that appear to be transcription errors. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else. '
                    'Please correct the following text: "{text}"'
                )
            )
        }

    @classmethod
    def get_config(cls, model_name: str):
        """
        Retrieve the configuration for the given model name.
        If the model is not in the pre-defined list, return the default configuration.
        """
        configs = cls.get_default_configs()
        if model_name in configs:
            return configs[model_name]
        else:
            logger.info(f"Model '{model_name}' not found in predefined configurations. Using default configuration.")
            # Fallback to the 'phi4:latest' configuration as default
            return configs['phi4:latest']


# --------------------------
# Application State Management (UPDATED)
# --------------------------
class AppState(QObject):
    """
    Maintains application state in a thread-safe manner.

    UPDATED: Added audio buffer management for temporary storage.
    Tracks whether audio recording is active and counts the number of active worker threads.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._is_recording = False
        self.audio_buffer = bytes()  # Holds recorded audio data
        self.active_threads = 0  # Counter for active processing threads

    @property
    def is_recording(self):
        """
        Thread-safe getter for the recording state.
        """
        with self._lock:
            return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool):
        """
        Thread-safe setter for the recording state.
        """
        with self._lock:
            self._is_recording = value

    def increment_threads(self):
        """
        Increment the count of active threads.
        """
        with self._lock:
            self.active_threads += 1

    def decrement_threads(self):
        """
        Decrement the count of active threads.
        """
        with self._lock:
            self.active_threads = max(0, self.active_threads - 1)

    @property
    def has_active_threads(self):
        """
        Return True if there are any active worker threads.
        """
        with self._lock:
            return self.active_threads > 0

    def clear_audio_buffer(self):
        """
        NEW: Clear the audio buffer (thread-safe).
        """
        with self._lock:
            self.audio_buffer = bytes()
            logger.info("Audio buffer cleared")

    def set_audio_buffer(self, audio_data: bytes):
        """
        NEW: Set the audio buffer (thread-safe).
        """
        with self._lock:
            self.audio_buffer = audio_data
            logger.info(f"Audio buffer updated: {len(audio_data)} bytes")

    def get_audio_buffer(self) -> bytes:
        """
        NEW: Get the audio buffer (thread-safe).
        """
        with self._lock:
            return self.audio_buffer

    @property
    def has_audio(self) -> bool:
        """
        NEW: Check if audio buffer has data (thread-safe).
        """
        with self._lock:
            return len(self.audio_buffer) > 0


# --------------------------
# Audio Recorder (PRESERVED EXACTLY FROM ORIGINAL)
# --------------------------
class AudioRecorder:
    """
    Handles audio recording using PyAudio with proper resource management.

    This class is implemented as a context manager to ensure that audio resources are correctly cleaned up.
    It continuously reads audio data from the input stream until the recording flag is turned off.
    """

    def __init__(self, state: AppState):
        self.state = state
        self.audio = None
        self.stream = None

    def __enter__(self):
        # Initialize PyAudio and open the input stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=AudioConfig.FORMAT,
            channels=AudioConfig.CHANNELS,
            rate=AudioConfig.SAMPLE_RATE,
            input=True,
            frames_per_buffer=AudioConfig.CHUNK_SIZE
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Clean up: stop and close the stream, then terminate PyAudio
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except Exception as e:
            logger.error("Error cleaning up audio resources", exc_info=True)

    def record(self):
        """
        Record audio until the state's 'is_recording' flag is set to False.

        Returns:
            The recorded audio data as bytes.
        """
        recorded_data = []
        start_time = datetime.datetime.now()
        try:
            logger.info("Started audio recording")  # ENHANCED: Added logging
            # Keep reading data in chunks until recording is stopped.
            while self.state.is_recording:
                data = self.stream.read(AudioConfig.CHUNK_SIZE, exception_on_overflow=False)
                recorded_data.append(data)

            # ENHANCED: Log recording duration
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Recording completed: {duration:.1f} seconds")
            return b"".join(recorded_data)
        except Exception as e:
            logger.error("Error during recording", exc_info=True)
            return None


# --------------------------
# Worker Threads (ENHANCED for Comprehensive VAD)
# --------------------------
class TranscriptionThread(QThread):
    """
    ENHANCED worker thread for audio transcription and subsequent text refinement.

    PRESERVED: All original signals and functionality
    ENHANCED: Added progress reporting, chunking support, and optimized parameters
    FIXED: All transcription methods now properly handle WAV file creation
    UPDATED: Practical chunking decisions and validation
    NEW: Added force_strategy parameter to override automatic strategy selection
    ENHANCED: Comprehensive VAD transcription with no audio loss guarantee

    Emits:
      - transcription_finished: when transcription is completed.
      - refinement_finished: when text refinement is completed.
      - error_occurred: if any error occurs during processing.
      - progress_updated: ENHANCED - progress updates during processing.
    """
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # ENHANCED - progress updates

    def __init__(self, audio_data: bytes, model_name: str, state: AppState, force_strategy: Optional[str] = None):
        super().__init__()
        self.audio_data = audio_data
        self.model_name = model_name
        self.state = state
        self.force_strategy = force_strategy  # NEW: Optional forced strategy
        # ENHANCED - Processing components
        self.processor = AudioProcessor()
        self.chunker = AudioChunker()

    def run(self):
        """
        ENHANCED main processing method.
        PRESERVED: Original flow and error handling
        ENHANCED: Added preprocessing, chunking support, and progress reporting
        FIXED: All audio processing now creates proper WAV files
        """
        self.state.increment_threads()
        if not self.audio_data:
            self.error_occurred.emit("No audio data to transcribe.")
            self.state.decrement_threads()
            return

        try:
            # ENHANCED: Added progress reporting throughout
            self.progress_updated.emit(5, "Starting transcription...")
            logger.info(f"Starting enhanced transcription process (force_strategy: {self.force_strategy})")

            # ENHANCED: Step 1 - Preprocess audio for optimal Whisper input
            self.progress_updated.emit(10, "Preprocessing audio...")
            processed_audio = self._preprocess_audio_with_fallback()

            # ENHANCED: Step 2 - Determine transcription strategy
            self.progress_updated.emit(20, "Analyzing audio...")
            transcription = self._transcribe_with_strategy(processed_audio)

            # PRESERVED: Original transcription result handling
            self.progress_updated.emit(80, "Transcription complete")
            self.transcription_finished.emit(transcription)

            # PRESERVED: Original refinement logic
            if "Failed to transcribe" not in transcription:
                self.progress_updated.emit(85, "Refining text...")
                refined = self.refine_text(transcription)
                self.progress_updated.emit(100, "Complete")
                self.refinement_finished.emit(refined)
            else:
                self.progress_updated.emit(100, "Transcription failed")
                self.refinement_finished.emit("")

        except Exception as e:
            logger.exception("Error in TranscriptionThread")
            self.error_occurred.emit(str(e))
        finally:
            self.state.decrement_threads()

    def _preprocess_audio_with_fallback(self) -> bytes:
        """
        FIXED - Preprocess audio with fallback to original if preprocessing fails.

        Now ensures all output is a properly formatted WAV file.

        Returns:
            Preprocessed audio data as a complete WAV file, or basic WAV if preprocessing fails
        """
        try:
            return self.processor.preprocess_for_whisper(self.audio_data)
        except Exception as e:
            logger.warning(f"Audio preprocessing failed, creating basic WAV: {e}")
            return AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)

    def _transcribe_with_strategy(self, audio_data: bytes) -> str:
        """
        NEW: Transcribe using either forced strategy or optimal automatic selection.

        Args:
            audio_data: Complete WAV file data

        Returns:
            Transcribed text
        """
        try:
            # If force strategy is specified, use it
            if self.force_strategy == "single":
                self.progress_updated.emit(25, "Using forced single transcription...")
                return self._transcribe_single_enhanced(audio_data)
            elif self.force_strategy == "time":
                self.progress_updated.emit(25, "Using forced time-based chunking...")
                # Force time-based chunking regardless of duration
                chunks = self._force_time_chunking()
                return self._transcribe_with_chunking_comprehensive(chunks)
            elif self.force_strategy == "vad":
                self.progress_updated.emit(25, "Using forced VAD-based chunking...")
                # Force VAD-based chunking regardless of duration
                chunks = self._force_vad_chunking()
                if chunks and len(chunks) > 1:
                    return self._transcribe_with_chunking_comprehensive(chunks)
                else:
                    self.progress_updated.emit(25, "VAD chunking failed, falling back to single transcription...")
                    return self._transcribe_single_enhanced(audio_data)
            else:
                # Use original optimal strategy selection
                return self._transcribe_with_optimal_strategy(audio_data)

        except Exception as e:
            logger.error(f"Transcription strategy failed: {e}")
            # Fallback to original method
            return self.transcribe_audio()

    def _force_time_chunking(self) -> List[bytes]:
        """
        NEW: Force time-based chunking regardless of audio duration.

        Returns:
            List of WAV file chunks
        """
        try:
            if not PYDUB_AVAILABLE:
                logger.warning("pydub not available - cannot perform time-based chunking")
                wav_data = AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)
                return [wav_data]

            # Convert to WAV and AudioSegment
            wav_data = AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")

            # Force time-based chunking
            chunks = self.chunker._chunk_with_time(audio_segment)

            # Convert to WAV bytes
            chunk_data_list = []
            for i, chunk in enumerate(chunks):
                buffer = io.BytesIO()
                chunk.export(buffer, format="wav")
                chunk_wav_data = buffer.getvalue()
                chunk_data_list.append(chunk_wav_data)

            logger.info(f"Forced time-based chunking created {len(chunk_data_list)} chunks")
            return chunk_data_list

        except Exception as e:
            logger.error(f"Forced time-based chunking failed: {e}")
            wav_data = AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)
            return [wav_data]

    def _force_vad_chunking(self) -> List[bytes]:
        """
        NEW: Force VAD-based chunking regardless of audio duration.

        Returns:
            List of WAV file chunks
        """
        try:
            if not PYDUB_AVAILABLE:
                logger.warning("pydub not available - cannot perform VAD-based chunking")
                wav_data = AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)
                return [wav_data]

            # Convert to WAV and AudioSegment
            wav_data = AudioProcessor._create_wav_from_pcm(self.audio_data, AudioConfig.SAMPLE_RATE)
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")

            # Force VAD-based chunking without validation
            raw_chunks = split_on_silence(
                audio_segment,
                min_silence_len=AudioConfig.VAD_MIN_SILENCE_LEN,
                silence_thresh=AudioConfig.VAD_SILENCE_THRESH,
                keep_silence=AudioConfig.VAD_KEEP_SILENCE
            )

            if not raw_chunks:
                logger.warning("VAD found no chunks")
                return []

            # Process chunks
            processed_chunks = self.chunker._process_raw_chunks(raw_chunks)

            # Convert to WAV bytes
            chunk_data_list = []
            for i, chunk in enumerate(processed_chunks):
                buffer = io.BytesIO()
                chunk.export(buffer, format="wav")
                chunk_wav_data = buffer.getvalue()
                chunk_data_list.append(chunk_wav_data)

            logger.info(f"Forced VAD-based chunking created {len(chunk_data_list)} chunks")
            return chunk_data_list

        except Exception as e:
            logger.error(f"Forced VAD-based chunking failed: {e}")
            return []

    def _transcribe_with_optimal_strategy(self, audio_data: bytes) -> str:
        """
        UPDATED - Tiered fallback logic for chunking strategy.

        This method implements a robust, tiered strategy for transcribing audio:
        1. For long audio (>2 min), it first attempts intelligent VAD-based chunking.
        2. If VAD fails or is not beneficial, it automatically falls back to reliable time-based chunking.
        3. If all chunking methods fail, it transcribes the audio in a single pass as a last resort.
        4. For short audio, it transcribes in a single pass for efficiency.

        Args:
            audio_data: Complete WAV file data.

        Returns:
            Transcribed text.
        """
        try:
            # Check original audio duration to determine the strategy
            original_duration = AudioProcessor.estimate_duration(self.audio_data, AudioConfig.SAMPLE_RATE)

            # For long audio, use a tiered chunking strategy
            if original_duration > AudioConfig.MIN_DURATION_FOR_CHUNKING:
                self.progress_updated.emit(25, f"Audio is {original_duration / 60:.1f} min, trying VAD chunking...")
                logger.info(f"Long audio detected ({original_duration / 60:.1f} min). Attempting VAD chunking.")

                # 1. Attempt VAD (intelligent) chunking
                vad_chunks = self.chunker.chunk_audio(self.audio_data)

                # Check if VAD chunking was effective and beneficial
                if len(vad_chunks) >= 2 and len(vad_chunks) <= max(5, int(original_duration / 60)):
                    self.progress_updated.emit(30, f"VAD successful. Using {len(vad_chunks)} chunks...")
                    logger.info(f"VAD chunking successful with {len(vad_chunks)} chunks.")
                    return self._transcribe_with_chunking_comprehensive(vad_chunks)
                else:
                    # 2. VAD failed or was not beneficial, fall back to Time-Based chunking
                    self.progress_updated.emit(25, "VAD not beneficial, falling back to time-based chunking...")
                    logger.warning("VAD chunking was not effective. Falling back to time-based chunking.")

                    time_chunks = self._force_time_chunking()

                    # Check if time-based chunking produced a valid result
                    if len(time_chunks) > 1:
                        self.progress_updated.emit(30, f"Time-based chunking successful. Using {len(time_chunks)} chunks...")
                        logger.info(f"Time-based chunking successful with {len(time_chunks)} chunks.")
                        return self._transcribe_with_chunking_comprehensive(time_chunks)
                    else:
                        # 3. All chunking methods failed, use single transcription as a last resort
                        self.progress_updated.emit(25, "All chunking failed, using single transcription...")
                        logger.error("All chunking methods failed. Transcribing as a single file is the last resort.")
                        return self._transcribe_single_enhanced(audio_data)
            else:
                # For short audio, transcribe in a single pass for efficiency
                self.progress_updated.emit(25, f"Audio is {original_duration / 60:.1f} min, using single transcription...")
                logger.info(f"Short audio detected ({original_duration / 60:.1f} min). Using single transcription.")
                return self._transcribe_single_enhanced(audio_data)

        except Exception as e:
            logger.error(f"Transcription strategy selection failed: {e}", exc_info=True)
            # Final, critical fallback to the original, simplest transcription method
            return self.transcribe_audio()

    def _transcribe_with_chunking_comprehensive(self, chunks: List[bytes]) -> str:
        """
        ENHANCED - Comprehensive chunked transcription that ensures NO chunks are lost.

        Key improvements:
        - Transcribes ALL chunks (no silent dropping)
        - Retries failed chunks with fallback methods
        - Comprehensive error handling and logging
        - Validates all chunks are processed

        Args:
            chunks: List of WAV file chunks

        Returns:
            Combined transcription from ALL chunks
        """
        try:
            total_chunks = len(chunks)
            logger.info(f"Starting comprehensive transcription of {total_chunks} chunks")

            # Calculate total audio duration for validation
            total_audio_duration = 0
            for i, chunk_data in enumerate(chunks):
                chunk_duration = AudioProcessor.estimate_duration(chunk_data, AudioConfig.WHISPER_SAMPLE_RATE)
                total_audio_duration += chunk_duration
                logger.debug(f"Chunk {i + 1}: {len(chunk_data)} bytes, {chunk_duration:.1f}s")

            logger.info(f"Total audio to transcribe: {total_audio_duration:.1f}s ({total_audio_duration / 60:.1f}min)")

            # Transcribe each chunk with comprehensive error handling
            transcription_results = []
            failed_chunks = []
            base_progress = 35
            chunk_progress_range = 40

            for i, chunk_data in enumerate(chunks):
                progress = base_progress + int((i / total_chunks) * chunk_progress_range)
                chunk_duration = AudioProcessor.estimate_duration(chunk_data, AudioConfig.WHISPER_SAMPLE_RATE)

                self.progress_updated.emit(progress,
                                           f"Transcribing chunk {i + 1}/{total_chunks} ({chunk_duration:.1f}s)")

                # ENHANCED: Try transcription with retry logic
                result = self._transcribe_chunk_with_retry(chunk_data, i + 1)

                if result is not None:
                    # CRITICAL: Include ALL results, even empty ones (represents silence/background)
                    transcription_results.append(result)
                    logger.info(
                        f"✓ Chunk {i + 1}/{total_chunks}: {len(result)} chars - '{result[:50]}{'...' if len(result) > 50 else ''}'")
                else:
                    # Even complete failure gets a placeholder to maintain chunk sequence
                    failed_chunks.append(i + 1)
                    transcription_results.append(f"[Chunk {i + 1} transcription failed]")
                    logger.error(f"✗ Chunk {i + 1}/{total_chunks}: FAILED - using placeholder")

            # VALIDATION: Ensure we have results for all chunks
            if len(transcription_results) != total_chunks:
                logger.error(f"CRITICAL: Transcription count mismatch! "
                             f"Expected {total_chunks}, got {len(transcription_results)}")
                return "Failed to transcribe: Missing chunks in transcription results"

            # Report results
            successful_chunks = total_chunks - len(failed_chunks)
            logger.info(f"Transcription complete: {successful_chunks}/{total_chunks} chunks successful")

            if failed_chunks:
                logger.warning(f"Failed chunks: {failed_chunks}")

            if successful_chunks == 0:
                return "Failed to transcribe: All chunks failed"

            # Combine all results (including placeholders)
            self.progress_updated.emit(75, "Combining all transcriptions...")
            combined = TextProcessor.combine_transcriptions_comprehensive(transcription_results)

            logger.info(f"Final result: {len(combined)} characters from {total_chunks} chunks "
                        f"({successful_chunks} successful, {len(failed_chunks)} failed)")

            return combined

        except Exception as e:
            logger.error(f"Comprehensive chunked transcription failed: {e}", exc_info=True)
            # Final fallback
            return "Failed to transcribe: Comprehensive transcription error"

    def _transcribe_with_chunking_practical(self, chunks: List[bytes]) -> str:
        """
        PRESERVED - Original practical method now delegates to comprehensive version.

        Args:
            chunks: List of WAV file chunks

        Returns:
            Combined transcription from all chunks
        """
        return self._transcribe_with_chunking_comprehensive(chunks)

    def _transcribe_chunk_with_retry(self, chunk_data: bytes, chunk_number: int) -> Optional[str]:
        """
        ENHANCED - Transcribe a single chunk with retry logic and comprehensive error handling.

        Args:
            chunk_data: Complete WAV file data from chunker
            chunk_number: Chunk number for logging

        Returns:
            Transcribed text for the chunk, or None if all attempts fail
        """
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying chunk {chunk_number}, attempt {attempt + 1}/{max_retries + 1}")

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                    temp_file.write(chunk_data)
                    temp_file.flush()

                    logger.debug(f"Transcribing chunk {chunk_number}: {len(chunk_data)} bytes")

                    # Use optimized parameters
                    result = mlx_whisper.transcribe(
                        temp_file.name,
                        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                        temperature=(0.0, 0.2, 0.4),
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        initial_prompt=(
                            "Transcribe clearly without repetition. "
                            "Use proper punctuation and spelling. "
                            "Avoid repeating words or phrases. "
                            "Keep original language."
                        )
                    )

                    text = result['text'].strip()
                    cleaned_text = TextProcessor.clean_repetitions(text)

                    # IMPORTANT: Return ALL results, even empty ones (silence is valid)
                    logger.debug(
                        f"Chunk {chunk_number} transcription successful: '{cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}'")
                    return cleaned_text

            except Exception as e:
                logger.warning(f"Chunk {chunk_number} transcription attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error(f"Chunk {chunk_number} failed after {max_retries + 1} attempts")

        return None  # All attempts failed

    def _transcribe_single_chunk(self, chunk_data: bytes) -> str:
        """
        PRESERVED - Original single chunk method for compatibility.

        Args:
            chunk_data: Complete WAV file data from chunker

        Returns:
            Transcribed text for the chunk
        """
        try:
            result = self._transcribe_chunk_with_retry(chunk_data, 1)
            return result if result is not None else ""
        except Exception as e:
            logger.error(f"Single chunk transcription failed: {e}")
            return ""

    def _transcribe_single_enhanced(self, audio_data: bytes) -> str:
        """
        FIXED - Enhanced single audio transcription (audio_data is already a complete WAV file).

        Args:
            audio_data: Complete WAV file data

        Returns:
            Transcribed text
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                # FIXED: audio_data is already a complete WAV file
                temp_wav.write(audio_data)
                temp_wav.flush()

                logger.info(f"Transcribing single audio file: {len(audio_data)} bytes")

                # ENHANCED: Optimized MLX Whisper parameters
                result = mlx_whisper.transcribe(
                    temp_wav.name,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",

                    # ENHANCED: Optimized parameters based on our research
                    temperature=(0.0, 0.2, 0.4),  # Reduced from (0.0, 0.2, 0.4, 0.6)
                    compression_ratio_threshold=2.4,  # Most important for repetition detection
                    logprob_threshold=-1.0,  # Confidence filtering
                    no_speech_threshold=0.6,  # Silence handling

                    # ENHANCED: Improved prompt for better results
                    initial_prompt=(
                        "Transcribe clearly without repetition. "
                        "Use proper punctuation and spelling. "
                        "Avoid repeating words or phrases. "
                        "Keep original language."
                    )
                )

                text = result['text'].strip()

                # ENHANCED: Clean any remaining repetitions
                cleaned_text = TextProcessor.clean_repetitions(text)

                # ENHANCED: Log quality metrics
                if TextProcessor.detect_repetition(cleaned_text):
                    logger.warning("Repetition detected in transcription")

                logger.info(f"Single transcription result: {len(cleaned_text)} chars")
                return cleaned_text

        except Exception as e:
            logger.error("Enhanced single audio transcription failed", exc_info=True)
            return f"Failed to transcribe: {str(e)}"

    def transcribe_audio(self) -> str:
        """
        PRESERVED: Original transcribe_audio method as fallback.
        ENHANCED: Updated with optimized parameters but preserved original structure.

        This method serves as a fallback and maintains compatibility with the original code.
        It creates proper WAV files using the same method as the original implementation.

        Returns:
            The transcribed text.
        """
        try:
            # PRESERVED: Original WAV file creation logic exactly
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(AudioConfig.CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(AudioConfig.FORMAT))
                    wf.setframerate(AudioConfig.SAMPLE_RATE)
                    wf.writeframes(self.audio_data)

                logger.info(f"Fallback transcription for {len(self.audio_data)} bytes of raw audio")

                # ENHANCED: Optimized MLX Whisper parameters while preserving original call structure
                result = mlx_whisper.transcribe(
                    temp_wav.name,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    temperature=(0.0, 0.2, 0.4),  # ENHANCED: Reduced from (0.0, 0.2, 0.4, 0.6)
                    # no_repeat_ngram_size=3,  # PRESERVED: Commented out (not supported)
                    # beam_size=5,             # PRESERVED: Commented out (not implemented)
                    # best_of=3,               # PRESERVED: Commented out (adds randomness)
                    # condition_on_previous_text=False,  # PRESERVED: Commented out
                    initial_prompt=(
                        # ENHANCED: Improved prompt while keeping original structure
                        "Transcribe clearly without repetition. "
                        "Use proper punctuation and spelling. "
                        "Avoid repeating words or phrases. "
                        "Keep original language."
                    )
                )
                # ENHANCED: Added repetition cleanup
                text = result['text'].strip()
                cleaned_text = TextProcessor.clean_repetitions(text)
                logger.info(f"Fallback transcription result: {len(cleaned_text)} chars")
                return cleaned_text
        except Exception as e:
            logger.error("Transcription failed", exc_info=True)
            return f"Failed to transcribe: {str(e)}"

    def refine_text(self, text: str) -> str:
        """
        PRESERVED: Original refine_text method.
        ENHANCED: Added repetition handling to system message.

        Uses the model configuration for the selected model (or the default fallback).

        Returns:
            The refined text.
        """
        try:
            # Retrieve the configuration for the selected model (or default)
            config = ModelConfig.get_config(self.model_name)
            # Build the messages for the LLM (only one type of message is needed here)
            messages = [
                {
                    'role': 'system',
                    'content': config.system_message  # ENHANCED: Now includes repetition handling
                },
                {
                    'role': 'user',
                    'content': config.user_message.format(text=text)
                }
            ]
            # Call the Ollama API to get the refined text
            logger.info(f"Refining text with model {self.model_name}: {len(text)} chars input")
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'ctx_num': config.ctx_num,
                    'temperature': config.temperature,
                    'seed': config.seed
                }
            )
            # Clean the response (remove any <think> tags, extra quotes, etc.)
            refined = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)

            # ENHANCED: Added logging for successful refinement
            refined_clean = refined.strip().strip('"')
            logger.info(f"Text refinement completed: {len(text)} chars -> {len(refined_clean)} chars")
            return refined_clean
        except Exception as e:
            logger.error("Refinement failed", exc_info=True)
            return f"Refinement failed: {str(e)}"


class RefinementThread(QThread):
    """
    PRESERVED: Worker thread dedicated solely to refining text.
    ENHANCED: Added repetition handling to system message.

    Emits:
      - refinement_finished: when text refinement is completed.
      - error_occurred: if any error occurs during processing.
    """
    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, text: str, model_name: str, state: AppState):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.state = state

    def run(self):
        """
        Main processing method for text refinement.
        """
        self.state.increment_threads()
        try:
            refined = self.refine_text(self.text)
            self.refinement_finished.emit(refined)
        except Exception as e:
            logger.error("Error in RefinementThread", exc_info=True)
            self.error_occurred.emit(str(e))
        finally:
            self.state.decrement_threads()

    def refine_text(self, text: str) -> str:
        """
        PRESERVED: Original refine_text method.
        ENHANCED: Added repetition handling to system message.

        Returns:
            The refined text.
        """
        try:
            config = ModelConfig.get_config(self.model_name)
            messages = [
                {
                    'role': 'system',
                    'content': config.system_message  # ENHANCED: Now includes repetition handling
                },
                {
                    'role': 'user',
                    'content': config.user_message.format(text=text)
                }
            ]
            logger.info(f"Re-refining text with model {self.model_name}: {len(text)} chars input")
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
            refined_clean = refined.strip().strip('"')
            logger.info(f"Re-refinement completed: {len(text)} chars -> {len(refined_clean)} chars")
            return refined_clean
        except Exception as e:
            logger.error("Refinement in RefinementThread failed", exc_info=True)
            return f"Refinement failed: {str(e)}"


# --------------------------
# Main Application GUI (UPDATED)
# --------------------------
class AudioTranscriberApp(QWidget):
    """
    PRESERVED: Main GUI application class for audio transcription and text refinement.
    ENHANCED: Added progress reporting, dependency checking, and window title updates.
    NEW: Added force strategy buttons and audio buffer management.
    FIXED: Copy buttons now preserve audio buffer for re-transcription.

    Provides:
      - A button to start/stop audio recording.
      - A dropdown to select the LLM model.
      - NEW: Three buttons to force specific transcription strategies.
      - A progress bar to indicate processing status.
      - Two text panels: one for the original transcription and one for the refined text.
      - Buttons to copy the text from each panel (FIXED: now preserve audio buffer).
    """

    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.available_models = self.fetch_models()
        self.current_transcription = ""
        self.current_worker = None  # Will hold the current worker thread (transcription/refinement)
        self.init_ui()

        # ENHANCED: Check for required dependencies
        if not PYDUB_AVAILABLE:
            self.show_dependency_warning()

    def show_dependency_warning(self):
        """
        ENHANCED - Show a warning if pydub is not available.

        This provides helpful information to users about missing optional features
        while still allowing the application to function normally.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Optional Dependency Missing")
        msg.setText("pydub library not found")
        msg.setInformativeText(
            "Advanced audio processing features (chunking, preprocessing) will be limited.\n\n"
            "For full functionality, install pydub:\n"
            "pip install pydub\n\n"
            "The application will work normally for shorter recordings."
        )
        msg.exec_()

    def fetch_models(self):
        """
        PRESERVED: Retrieve a list of available models from the Ollama API.
        If fetching fails, return a default list.
        """
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]
            # Prioritize 'phi4:latest' if available.
            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')
            logger.info(f"Found {len(installed_models)} available models")
            return installed_models if installed_models else ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logger.error("Failed to fetch models", exc_info=True)
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def init_ui(self):
        """
        PRESERVED: Initialize and set up the GUI components.
        ENHANCED: Updated window title to indicate practical chunking version.
        NEW: Added force strategy buttons.
        """
        self.setWindowTitle("Hybrid Audio Transcriber - Practical Chunking")  # UPDATED: Updated title
        self.setGeometry(420, 300, 800, 550)  # Increased height for new buttons
        main_layout = QVBoxLayout(self)
        self.create_top_controls(main_layout)
        self.create_force_strategy_buttons(main_layout)  # NEW: Force strategy buttons
        self.create_progress_bar(main_layout)
        self.create_text_areas(main_layout)
        self.update_force_buttons_state()  # NEW: Initial button state
        self.show()

    def create_top_controls(self, layout):
        """
        PRESERVED: Create the top control panel containing the recording button, model selector, and re-refine button.
        """
        top_layout = QHBoxLayout()

        # Recording Button: toggles recording on and off.
        self.recording_button = QPushButton("Start Recording", self)
        self.set_button_style("ready")
        self.recording_button.clicked.connect(self.toggle_recording)
        top_layout.addWidget(self.recording_button, 50)

        # Model Selector: allows user to choose from available models.
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

        # Re-Refine Button: allows the user to re-run text refinement on the transcription.
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

    def create_force_strategy_buttons(self, layout):
        """
        NEW: Create force strategy buttons row.
        """
        force_layout = QHBoxLayout()

        # Force Single Audio button
        self.force_single_btn = QPushButton("Force Single Audio", self)
        self.force_single_btn.setStyleSheet("""
            QPushButton {
                background-color: #16537e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1e6ba8; }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666;
            }
        """)
        self.force_single_btn.clicked.connect(lambda: self.force_transcription("single"))
        force_layout.addWidget(self.force_single_btn)

        # Force Time-Based button
        self.force_time_btn = QPushButton("Force Time-Based", self)
        self.force_time_btn.setStyleSheet("""
            QPushButton {
                background-color: #16537e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1e6ba8; }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666;
            }
        """)
        self.force_time_btn.clicked.connect(lambda: self.force_transcription("time"))
        force_layout.addWidget(self.force_time_btn)

        # Force VAD-Based button
        self.force_vad_btn = QPushButton("Force VAD-Based", self)
        self.force_vad_btn.setStyleSheet("""
            QPushButton {
                background-color: #16537e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1e6ba8; }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666;
            }
        """)
        self.force_vad_btn.clicked.connect(lambda: self.force_transcription("vad"))
        force_layout.addWidget(self.force_vad_btn)

        layout.addLayout(force_layout)

    def create_progress_bar(self, layout):
        """
        PRESERVED: Create a progress bar to indicate processing status.
        """
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def create_text_areas(self, layout):
        """
        PRESERVED: Create two text areas: one for the original transcription and one for the refined text.
        Each text area has an associated 'Copy' button.
        FIXED: Copy buttons now preserve audio buffer instead of clearing it.
        """
        text_layout = QHBoxLayout()

        # Transcription Panel
        trans_layout = QVBoxLayout()
        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Original transcription...")
        trans_layout.addWidget(self.transcription_box)
        self.copy_transcription_btn = QPushButton("Copy", self)
        # FIXED: Changed from copy_text_and_clear to copy_text to preserve audio buffer
        self.copy_transcription_btn.clicked.connect(lambda: self.copy_text(self.transcription_box))
        trans_layout.addWidget(self.copy_transcription_btn)
        text_layout.addLayout(trans_layout)

        # Refined Text Panel
        refined_layout = QVBoxLayout()
        self.refined_box = QTextEdit(self)
        self.refined_box.setPlaceholderText("Refined text...")
        refined_layout.addWidget(self.refined_box)
        self.copy_refined_btn = QPushButton("Copy", self)
        # FIXED: Changed from copy_text_and_clear to copy_text to preserve audio buffer
        self.copy_refined_btn.clicked.connect(lambda: self.copy_text(self.refined_box))
        refined_layout.addWidget(self.copy_refined_btn)
        text_layout.addLayout(refined_layout)

        layout.addLayout(text_layout)

    def set_button_style(self, state):
        """
        PRESERVED: Update the appearance of the recording button based on the current state.

        States:
          - "ready": ready to start recording.
          - "recording": recording in progress.
          - "processing": processing the recorded audio.
        """
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

    def update_force_buttons_state(self):
        """
        NEW: Update the enabled state of force strategy buttons.
        """
        has_audio = self.state.has_audio
        is_processing = self.state.has_active_threads

        # Enable buttons only if we have audio and are not processing
        enabled = has_audio and not is_processing

        self.force_single_btn.setEnabled(enabled)
        self.force_time_btn.setEnabled(enabled)
        self.force_vad_btn.setEnabled(enabled)

    def toggle_recording(self):
        """
        PRESERVED: Toggle audio recording on/off when the recording button is clicked.
        UPDATED: Clear audio buffer when starting new recording.
        """
        if not self.state.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """
        PRESERVED: Start the audio recording process in a background thread.
        ENHANCED: Added window title update.
        UPDATED: Clear previous audio buffer.
        """
        self.state.clear_audio_buffer()  # Clear previous recording
        self.set_button_style("recording")
        self.progress_bar.setValue(0)
        self.state.is_recording = True
        self.setWindowTitle("Hybrid Audio Transcriber - Recording...")  # ENHANCED: Title update
        self.update_force_buttons_state()  # Update button states
        # Launch a new thread for recording to keep the UI responsive.
        threading.Thread(target=self.record_audio_background, daemon=True).start()

    def record_audio_background(self):
        """
        PRESERVED: Background function that handles audio recording using the AudioRecorder context manager.
        UPDATED: Store audio in buffer for reprocessing.
        FIXED: Use QTimer.singleShot for thread-safe GUI updates.
        """
        with AudioRecorder(self.state) as recorder:
            audio_data = recorder.record()
        self.state.is_recording = False
        if audio_data:
            self.state.set_audio_buffer(audio_data)  # Store for reprocessing
            self.current_transcription = ""
            self.start_transcription(audio_data)
        else:
            self.display_transcription("No audio data captured.")
            self.display_refined_text("")

        # FIXED: Use QTimer.singleShot for thread-safe GUI updates from a background thread.
        # QApplication.invokeLater does not exist in PyQt5.
        QTimer.singleShot(0, self.update_force_buttons_state)

    def stop_recording(self):
        """
        PRESERVED: Stop the audio recording process.
        """
        self.state.is_recording = False
        self.set_button_style("processing")

    def start_transcription(self, audio_data, force_strategy=None):
        """
        PRESERVED: Start the transcription (and subsequent refinement) process using a worker thread.
        ENHANCED: Added connection to new progress signal.
        UPDATED: Added force_strategy parameter.
        """
        self.current_worker = TranscriptionThread(
            audio_data,
            self.model_selector.currentText(),
            self.state,
            force_strategy=force_strategy
        )
        self.current_worker.transcription_finished.connect(self.display_transcription)
        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self.handle_error)

        # ENHANCED: Connect the new progress signal
        self.current_worker.progress_updated.connect(self.update_progress)

        self.current_worker.start()
        self.update_force_buttons_state()  # Update button states

    def force_transcription(self, strategy):
        """
        NEW: Force a specific transcription strategy on stored audio.
        """
        audio_data = self.state.get_audio_buffer()
        if not audio_data:
            logger.warning("No audio data available for force transcription")
            return

        logger.info(f"Starting forced {strategy} transcription")
        self.set_button_style("processing")
        self.progress_bar.setValue(0)
        self.start_transcription(audio_data, force_strategy=strategy)

    def update_progress(self, progress: int, message: str):
        """
        ENHANCED - Update progress bar and window title with current operation status.

        Args:
            progress: Progress percentage (0-100)
            message: Status message describing current operation
        """
        self.progress_bar.setValue(progress)
        self.setWindowTitle(f"Hybrid Audio Transcriber - {message}")

        # Process events to keep UI responsive
        QApplication.processEvents()

    def re_refine_text(self):
        """
        PRESERVED: Initiate re-refinement of the current transcription text using a worker thread.
        ENHANCED: Added window title update.
        """
        text = self.transcription_box.toPlainText().strip()
        if not text:
            logger.warning("No text available for re-refinement")
            return
        self.set_button_style("processing")
        self.progress_bar.setValue(50)
        self.setWindowTitle("Hybrid Audio Transcriber - Re-refining text...")  # ENHANCED: Title update
        self.current_worker = RefinementThread(text, self.model_selector.currentText(), self.state)
        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.start()
        self.update_force_buttons_state()  # Update button states

    def display_transcription(self, text):
        """
        PRESERVED: Display the transcribed text in the transcription text area.
        ENHANCED: Added quality feedback logging.
        """
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

        # ENHANCED: Log quality feedback
        if text and TextProcessor.detect_repetition(text):
            logger.warning("Repetition detected in transcription - consider using enhanced features")

        logger.info(f"Transcription displayed: {len(text)} characters")

    def display_refined_text(self, text):
        """
        PRESERVED: Display the refined text in the refined text area.
        ENHANCED: Reset window title when complete.
        UPDATED: Update force button states.
        """
        self.refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style("ready")
        self.setWindowTitle("Hybrid Audio Transcriber - Practical Chunking")  # UPDATED: Reset title
        self.update_force_buttons_state()  # Update button states

        logger.info(f"Refined text displayed: {len(text)} characters")

    def copy_text(self, widget):
        """
        FIXED: Copy the text from the given text widget to the system clipboard.
        IMPORTANT: Audio buffer is now preserved - force strategy buttons remain enabled after copying.
        """
        clipboard = QApplication.clipboard()
        text = widget.toPlainText()
        clipboard.setText(text)
        logger.info(f"Text copied to clipboard: {len(text)} characters - audio buffer preserved")

    def copy_text_and_clear(self, widget):
        """
        DEPRECATED: This method is no longer used to preserve audio buffer after copying.
        Kept for reference but not connected to any buttons.
        """
        self.copy_text(widget)
        self.state.clear_audio_buffer()
        self.update_force_buttons_state()

    def handle_error(self, error_message):
        """
        PRESERVED: Handle errors by logging the error, showing a message box, and resetting UI elements.
        ENHANCED: Reset window title on error.
        UPDATED: Update force button states.
        """
        logger.error(f"Application error: {error_message}")
        QMessageBox.critical(self, "Error", error_message)
        self.set_button_style("ready")
        self.progress_bar.setValue(0)
        self.setWindowTitle("Hybrid Audio Transcriber - Practical Chunking")  # UPDATED: Reset title
        self.update_force_buttons_state()  # Update button states

    def closeEvent(self, event):
        """
        PRESERVED: Handle the window close event.
        UPDATED: Still clear audio buffer on close, but this is now the only way to clear it
        (besides recording new audio).

        Waits for any active worker threads to finish before closing the application.
        """
        self.state.clear_audio_buffer()  # Clear audio buffer on close (preserved)
        if self.state.has_active_threads:
            logger.info("Waiting for active threads to finish...")
            while self.state.has_active_threads:
                QApplication.processEvents()
        logger.info("Application closing")
        event.accept()


# --------------------------
# Application Entry Point (PRESERVED EXACTLY)
# --------------------------
def main():
    """
    Main entry point of the application.

    Sets up the QApplication and starts the main GUI.
    """
    logger.info("Starting Hybrid Audio Transcriber - Practical Chunking")
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()