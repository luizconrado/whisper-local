#!/usr/bin/env python3
"""
Hybrid Audio Transcriber Application
======================================
This application records audio, transcribes it using MLX Whisper, and refines (corrects) the text via an LLM (Ollama).
It combines robust configuration, resource management, error handling, and a clean PyQt5 GUI.

Key Features:
- No limitation on recording duration.
- Audio is automatically chunked for long recordings to improve transcription accuracy.
- No limitation on text length passed to the LLM.
- Uses a fallback configuration if the selected model is not in the pre-defined list.
- Extensive inline documentation and comments for clarity.
"""

import sys
import re
import logging
import threading
import datetime
import tempfile
import wave
from dataclasses import dataclass
from functools import partial
from contextlib import contextmanager

import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QThread, QObject

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
# Audio Configuration
# --------------------------
class AudioConfig:
    """
    Audio configuration settings.

    NOTE: We do not impose any upper limit on recording duration.
    """
    SAMPLE_RATE = 44100  # Samples per second
    FORMAT = pyaudio.paInt16  # 16-bit format
    CHANNELS = 1  # Mono recording
    CHUNK_SIZE = 1024  # Number of frames per buffer


# --------------------------
# Model Configuration
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
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
                user_message='"{text}"'
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
# Application State Management
# --------------------------
class AppState(QObject):
    """
    Maintains application state in a thread-safe manner.

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


# --------------------------
# Audio Recorder
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
            # Keep reading data in chunks until recording is stopped.
            while self.state.is_recording:
                data = self.stream.read(AudioConfig.CHUNK_SIZE, exception_on_overflow=False)
                recorded_data.append(data)
            return b"".join(recorded_data)
        except Exception as e:
            logger.error("Error during recording", exc_info=True)
            return None


# --------------------------
# Worker Threads
# --------------------------
class TranscriptionThread(QThread):
    """
    Worker thread for audio transcription and subsequent text refinement.
    For long recordings, it chunks the audio to improve transcription reliability.

    Emits:
      - transcription_finished: when transcription is completed.
      - refinement_finished: when text refinement is completed.
      - error_occurred: if any error occurs during processing.
    """
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, audio_data: bytes, model_name: str, state: AppState):
        super().__init__()
        self.audio_data = audio_data
        self.model_name = model_name
        self.state = state

    def run(self):
        """
        Main processing method.
        Transcribes the recorded audio and then refines the transcription.
        """
        self.state.increment_threads()
        if not self.audio_data:
            self.error_occurred.emit("No audio data to transcribe.")
            self.state.decrement_threads()
            return

        try:
            # Transcribe the audio data using MLX Whisper
            transcription = self.transcribe_audio()
            self.transcription_finished.emit(transcription)
            # If transcription succeeded, refine (correct) the text using the LLM
            if "Failed to transcribe" not in transcription and "Transcription resulted in no text" not in transcription:
                refined = self.refine_text(transcription)
                self.refinement_finished.emit(refined)
            else:
                self.refinement_finished.emit("")
        except Exception as e:
            logger.exception("Error in TranscriptionThread")
            self.error_occurred.emit(str(e))
        finally:
            self.state.decrement_threads()

    def transcribe_audio(self) -> str:
        """
        Transcribes the recorded audio to text by breaking it into overlapping chunks.

        The audio is first saved as a single WAV file. It is then segmented into
        30-second chunks. Each chunk is transcribed individually, and the transcription
        from the previous chunk is used as a prompt for the next, ensuring continuity
        and preventing repetitions.

        Returns:
            The full transcribed text, or an error message if transcription fails.
        """
        CHUNK_DURATION_S = 30  # Whisper works best with chunks of 30s or less.

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav_file:
            main_wav_path = temp_wav_file.name
            try:
                # 1. Save the entire audio recording to a single WAV file
                with wave.open(main_wav_path, 'wb') as wf:
                    wf.setnchannels(AudioConfig.CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(AudioConfig.FORMAT))
                    wf.setframerate(AudioConfig.SAMPLE_RATE)
                    wf.writeframes(self.audio_data)
                logger.info(f"Full audio saved to temporary file: {main_wav_path}")

                # 2. Read the main WAV file to prepare for chunking
                with wave.open(main_wav_path, 'rb') as wf:
                    frame_rate = wf.getframerate()
                    num_frames = wf.getnframes()
                    params = wf.getparams()
                    total_duration = num_frames / float(frame_rate)
                    logger.info(f"Audio duration: {total_duration:.2f} seconds.")

                    # If audio is short, transcribe it in one go
                    if total_duration <= CHUNK_DURATION_S:
                        logger.info("Audio is short, transcribing directly.")
                        result = mlx_whisper.transcribe(
                            main_wav_path,
                            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                            initial_prompt="Accurate transcript with proper punctuation."
                        )
                        return result['text'].strip()

                    # 3. Process audio in chunks
                    chunk_duration_frames = CHUNK_DURATION_S * frame_rate
                    full_transcription = []
                    previous_transcription_text = ""
                    start_frame = 0
                    part_number = 1

                    while start_frame < num_frames:
                        end_frame = min(start_frame + chunk_duration_frames, num_frames)
                        logger.info(f"Processing chunk {part_number}: frames {start_frame} to {end_frame}")

                        wf.setpos(start_frame)
                        chunk_data = wf.readframes(end_frame - start_frame)

                        with tempfile.NamedTemporaryFile(suffix=f'_part{part_number}.wav', delete=True) as chunk_wav_file:
                            with wave.open(chunk_wav_file.name, 'wb') as chunk_wf:
                                chunk_wf.setparams(params)
                                chunk_wf.writeframes(chunk_data)

                            try:
                                # Use the previous transcription as a prompt for the current chunk
                                result = mlx_whisper.transcribe(
                                    chunk_wav_file.name,
                                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                                    initial_prompt=previous_transcription_text
                                )
                                current_transcription = result['text'].strip()
                                full_transcription.append(current_transcription)
                                
                                # Update the prompt for the next iteration
                                previous_transcription_text += " " + current_transcription
                                logger.info(f"Chunk {part_number} transcribed successfully.")

                            except Exception as e:
                                logger.error(f"Failed to transcribe chunk {part_number}", exc_info=True)
                                full_transcription.append(f"[Error transcribing part {part_number}]")

                        start_frame += chunk_duration_frames
                        part_number += 1

                    return " ".join(full_transcription).strip()

            except Exception as e:
                logger.error("Transcription failed during chunking process", exc_info=True)
                return f"Failed to transcribe: {str(e)}"

    def refine_text(self, text: str) -> str:
        """
        Refine (correct) the transcribed text using an LLM.

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
                    'content': config.system_message
                },
                {
                    'role': 'user',
                    'content': config.user_message.format(text=text)
                }
            ]
            # Call the Ollama API to get the refined text
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
            return refined.strip().strip('"')
        except Exception as e:
            logger.error("Refinement failed", exc_info=True)
            return f"Refinement failed: {str(e)}"


class RefinementThread(QThread):
    """
    Worker thread dedicated solely to refining text.

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
        Refine (correct) the provided text using an LLM.

        Returns:
            The refined text.
        """
        try:
            config = ModelConfig.get_config(self.model_name)
            messages = [
                {
                    'role': 'system',
                    'content': config.system_message
                },
                {
                    'role': 'user',
                    'content': config.user_message.format(text=text)
                }
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
            return refined.strip().strip('"')
        except Exception as e:
            logger.error("Refinement in RefinementThread failed", exc_info=True)
            return f"Refinement failed: {str(e)}"


# --------------------------
# Main Application GUI
# --------------------------
class AudioTranscriberApp(QWidget):
    """
    Main GUI application class for audio transcription and text refinement.

    Provides:
      - A button to start/stop audio recording.
      - A dropdown to select the LLM model.
      - A progress bar to indicate processing status.
      - Two text panels: one for the original transcription and one for the refined text.
      - Buttons to copy the text from each panel.
    """

    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.available_models = self.fetch_models()
        self.current_transcription = ""
        self.current_worker = None  # Will hold the current worker thread (transcription/refinement)
        self.init_ui()

    def fetch_models(self):
        """
        Retrieve a list of available models from the Ollama API.
        If fetching fails, return a default list.
        """
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]
            # Prioritize 'phi4:latest' if available.
            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')
            return installed_models if installed_models else ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logger.error("Failed to fetch models", exc_info=True)
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def init_ui(self):
        """
        Initialize and set up the GUI components.
        """
        self.setWindowTitle("Hybrid Audio Transcriber")
        self.setGeometry(420, 300, 800, 500)
        main_layout = QVBoxLayout(self)
        self.create_top_controls(main_layout)
        self.create_progress_bar(main_layout)
        self.create_text_areas(main_layout)
        self.show()

    def create_top_controls(self, layout):
        """
        Create the top control panel containing the recording button, model selector, and re-refine button.
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

    def create_progress_bar(self, layout):
        """
        Create a progress bar to indicate processing status.
        """
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def create_text_areas(self, layout):
        """
        Create two text areas: one for the original transcription and one for the refined text.
        Each text area has an associated 'Copy' button.
        """
        text_layout = QHBoxLayout()

        # Transcription Panel
        trans_layout = QVBoxLayout()
        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Original transcription...")
        trans_layout.addWidget(self.transcription_box)
        self.copy_transcription_btn = QPushButton("Copy", self)
        self.copy_transcription_btn.clicked.connect(partial(self.copy_text, self.transcription_box))
        trans_layout.addWidget(self.copy_transcription_btn)
        text_layout.addLayout(trans_layout)

        # Refined Text Panel
        refined_layout = QVBoxLayout()
        self.refined_box = QTextEdit(self)
        self.refined_box.setPlaceholderText("Refined text...")
        refined_layout.addWidget(self.refined_box)
        self.copy_refined_btn = QPushButton("Copy", self)
        self.copy_refined_btn.clicked.connect(partial(self.copy_text, self.refined_box))
        refined_layout.addWidget(self.copy_refined_btn)
        text_layout.addLayout(refined_layout)

        layout.addLayout(text_layout)

    def set_button_style(self, state):
        """
        Update the appearance of the recording button based on the current state.

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

    def toggle_recording(self):
        """
        Toggle audio recording on/off when the recording button is clicked.
        """
        if not self.state.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """
        Start the audio recording process in a background thread.
        """
        self.set_button_style("recording")
        self.progress_bar.setValue(0)
        self.state.is_recording = True
        # Launch a new thread for recording to keep the UI responsive.
        threading.Thread(target=self.record_audio_background, daemon=True).start()

    def record_audio_background(self):
        """
        Background function that handles audio recording using the AudioRecorder context manager.
        """
        with AudioRecorder(self.state) as recorder:
            audio_data = recorder.record()
        self.state.is_recording = False
        if audio_data:
            self.current_transcription = ""
            self.start_transcription(audio_data)
        else:
            self.display_transcription("No audio data captured.")
            self.display_refined_text("")

    def stop_recording(self):
        """
        Stop the audio recording process.
        """
        self.state.is_recording = False
        self.set_button_style("processing")

    def start_transcription(self, audio_data):
        """
        Start the transcription (and subsequent refinement) process using a worker thread.
        """
        self.current_worker = TranscriptionThread(audio_data, self.model_selector.currentText(), self.state)
        self.current_worker.transcription_finished.connect(self.display_transcription)
        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.start()

    def re_refine_text(self):
        """
        Initiate re-refinement of the current transcription text using a worker thread.
        """
        text = self.transcription_box.toPlainText().strip()
        if not text:
            return
        self.set_button_style("processing")
        self.progress_bar.setValue(50)
        self.current_worker = RefinementThread(text, self.model_selector.currentText(), self.state)
        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.start()

    def display_transcription(self, text):
        """
        Display the transcribed text in the transcription text area.
        """
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

    def display_refined_text(self, text):
        """
        Display the refined text in the refined text area.
        """
        self.refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style("ready")

    def copy_text(self, widget):
        """
        Copy the text from the given text widget to the system clipboard.
        """
        clipboard = QApplication.clipboard()
        clipboard.setText(widget.toPlainText())

    def handle_error(self, error_message):
        """
        Handle errors by logging the error, showing a message box, and resetting UI elements.
        """
        logger.error(error_message)
        QMessageBox.critical(self, "Error", error_message)
        self.set_button_style("ready")
        self.progress_bar.setValue(0)

    def closeEvent(self, event):
        """
        Handle the window close event.

        Waits for any active worker threads to finish before closing the application.
        """
        if self.state.has_active_threads:
            logger.info("Waiting for active threads to finish...")
            while self.state.has_active_threads:
                QApplication.processEvents()
        event.accept()


# --------------------------
# Application Entry Point
# --------------------------
def main():
    """
    Main entry point of the application.

    Sets up the QApplication and starts the main GUI.
    """
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()