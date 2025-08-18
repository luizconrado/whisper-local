"""
Audio Transcriber Application with LLM Integration
Version 2.1 - Enhanced Stability and Code Quality Edition

Features:
- Audio recording and transcription using MLX Whisper
- LLM-powered text refinement via Ollama
- Model-specific configuration system
- Thread-safe resource management
- Enhanced error handling and logging
"""

import sys
import re
import logging
import threading
import datetime
import tempfile
import wave
import pyaudio
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox
)
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import mlx_whisper
import ollama

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcriber.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Audio Configuration
# --------------------------
class AudioConfig:
    SAMPLE_RATE = 44100
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_SIZE = 1024
    MAX_RECORD_SECONDS = 600

# --------------------------
# Model Configuration System
# --------------------------
class ModelConfig:
    """Encapsulates model-specific parameters and templates"""
    _DEFAULT = {
        'ctx_num': 8192,
        'temperature': 0.3,
        'seed': 42,
        'system_template': (
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
        'user_template': 'Correct this text: {text}',
        'max_input_length': 8192
    }

    _CONFIG = {
        'phi4:latest': {
            'ctx_num': 8192,
            'temperature': 0.2,
            'seed': 1,
            'system_template': (
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
            'user_template': 'Here is the text to be corrected: "{text}"',
            'max_input_length': 8192
        },
        'deepseek-r1:1.5b': {
            'ctx_num': 8192,
            'temperature': 1.3,
            'system_template': (
                'You are a helpful and professional text corrector.'
            ),
            'user_template': (
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                    'Here is the text to be corrected: "{text}"'
                ),
            'max_input_length': 8192
        }
    }

    @classmethod
    def get_config(cls, model_name: str) -> Dict[str, Any]:
        """Retrieve configuration for specified model with fallback to defaults"""
        return cls._CONFIG.get(model_name, cls._DEFAULT)

# --------------------------
# Application State Management
# --------------------------
class AppState(QObject):
    """Thread-safe state management for recording and processing"""
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._is_recording = False
        self._audio_buffer = bytes()
        self._active_threads = 0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool) -> None:
        with self._lock:
            self._is_recording = value

    @property
    def audio_buffer(self) -> bytes:
        with self._lock:
            return self._audio_buffer

    @audio_buffer.setter
    def audio_buffer(self, value: bytes) -> None:
        with self._lock:
            self._audio_buffer = value

    def increment_threads(self) -> None:
        with self._lock:
            self._active_threads += 1

    def decrement_threads(self) -> None:
        with self._lock:
            self._active_threads = max(0, self._active_threads - 1)

    @property
    def has_active_threads(self) -> bool:
        with self._lock:
            return self._active_threads > 0

# --------------------------
# Audio Recording Functions
# --------------------------
class AudioRecorder:
    """Handles audio recording with resource management"""
    def __init__(self, state: AppState):
        self.state = state
        self.audio = None
        self.stream = None

    def __enter__(self) -> 'AudioRecorder':
        self.audio = pyaudio.PyAudio()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Ensure proper resource cleanup"""
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except Exception as e:
            logger.error(f"Resource cleanup error: {e}")

    def record_audio(self) -> None:
        """Main recording loop with error handling"""
        try:
            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                frames_per_buffer=AudioConfig.CHUNK_SIZE
            )

            recorded_data = []
            while self.state.is_recording:
                data = self.stream.read(AudioConfig.CHUNK_SIZE, exception_on_overflow=False)
                recorded_data.append(data)

                if len(recorded_data) > (AudioConfig.SAMPLE_RATE / AudioConfig.CHUNK_SIZE * AudioConfig.MAX_RECORD_SECONDS):
                    logger.warning("Maximum recording time reached")
                    break

            self.state.audio_buffer = b''.join(recorded_data)
        except IOError as e:
            logger.error(f"Audio device I/O error: {e}")
            self.state.audio_buffer = None
        except Exception as e:
            logger.exception("Unexpected recording error")
            self.state.audio_buffer = None

# --------------------------
# Core Processing Threads
# --------------------------
class TranscriptionThread(QThread):
    """
    Handles audio transcription and initial refinement

    Signals:
        transcription_finished: str - Emits raw transcribed text
        refinement_finished: str - Emits refined text
    """
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)

    def __init__(self, audio_data: bytes, model_name: str, state: AppState):
        super().__init__()
        self.audio_data = audio_data
        self.model_name = model_name
        self.state = state

    def run(self) -> None:
        """Main processing pipeline"""
        try:
            self.state.increment_threads()

            if not self.audio_data:
                self._emit_error("No audio data received")
                return

            transcription_result = self.transcribe_audio()
            self.transcription_finished.emit(transcription_result)

            if "Failed to transcribe" not in transcription_result:
                self._handle_refinement(transcription_result)
            else:
                self.refinement_finished.emit("")

        finally:
            self.state.decrement_threads()

    def transcribe_audio(self) -> str:
        """Convert audio to text using Whisper"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                self._write_wav_file(temp_wav.name)
                result = mlx_whisper.transcribe(
                    temp_wav.name,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    initial_prompt=(
                        "Accurate transcript with proper punctuation. "
                        "Do not repeat phrases. Use standard spelling."
                    )
                )
                return result['text'].strip()
        except wave.Error as e:
            logger.error(f"WAV file error: {e}")
            return "Audio format error"
        except mlx_whisper.WhisperException as e:
            logger.error(f"Transcription error: {e}")
            return "Transcription service error"
        except Exception as e:
            logger.exception("Transcription failed")
            return f"Failed to transcribe: {str(e)}"

    def _write_wav_file(self, filename: str) -> None:
        """Helper for writing WAV files with validation"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(AudioConfig.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(AudioConfig.FORMAT))
            wf.setframerate(AudioConfig.SAMPLE_RATE)
            wf.writeframes(self.audio_data)

    def _handle_refinement(self, text: str) -> None:
        """Handle text refinement workflow"""
        try:
            refined = refine_text_with_model(text, self.model_name)
            self.refinement_finished.emit(refined)
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            self.refinement_finished.emit(f"Refinement failed: {str(e)}")

    def _emit_error(self, message: str) -> None:
        """Consistent error emission"""
        self.transcription_finished.emit(message)
        self.refinement_finished.emit("")

class RefinementThread(QThread):
    """Handles text refinement process"""
    refinement_finished = pyqtSignal(str)

    def __init__(self, text: str, model_name: str, state: AppState):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.state = state

    def run(self) -> None:
        """Main refinement execution"""
        try:
            self.state.increment_threads()

            if not self.text.strip():
                self.refinement_finished.emit("No text to refine")
                return

            if len(self.text) > ModelConfig.get_config(self.model_name)['max_input_length']:
                self.refinement_finished.emit("Text too long for refinement")
                return

            refined = refine_text_with_model(self.text, self.model_name)
            self.refinement_finished.emit(refined)

        except Exception as e:
            logger.error(f"Refinement error: {e}")
            self.refinement_finished.emit(f"Refinement failed: {str(e)}")
        finally:
            self.state.decrement_threads()

# --------------------------
# Core Processing Functions
# --------------------------
def refine_text_with_model(text: str, model_name: str) -> str:
    """
    Perform text refinement using specified LLM model

    Args:
        text: Input text to refine
        model_name: Name of model to use

    Returns:
        Refined text or error message
    """
    try:
        config = ModelConfig.get_config(model_name)
        messages = _create_messages(text, config)

        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                'ctx_num': config['ctx_num'],
                'temperature': config['temperature'],
                'seed': config.get('seed', 42)
            }
        )

        return _clean_response(response['message']['content'])
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e.error}")
        return f"API error: {e.error}"
    except Exception as e:
        logger.exception("Refinement failed")
        return f"Refinement error: {str(e)}"

def _create_messages(text: str, config: Dict) -> List[Dict]:
    """Create message payload with sanitized inputs"""
    sanitized_text = re.sub(r'[^\w\s.,?!-]', '', text)[:config['max_input_length']]
    return [
        {
            'role': 'system',
            'content': config['system_template'][:1000]  # Prevent prompt flooding
        },
        {
            'role': 'user',
            'content': config['user_template'].format(text=sanitized_text)
        }
    ]

def _clean_response(text: str) -> str:
    """Clean LLM response output"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip().strip('"')

# --------------------------
# Main Application GUI
# --------------------------
class AudioTranscriberApp(QWidget):
    """Main application window with GUI components"""
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.available_models = self.fetch_models()
        self.init_ui()
        self._processing_lock = threading.Lock()

    def fetch_models(self) -> List[str]:
        """Retrieve installed Ollama models"""
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]

            # Prioritize phi4 if available
            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')

            return installed_models or ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logger.error(f"Model fetch error: {e}")
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def init_ui(self) -> None:
        """Initialize user interface components"""
        self.setWindowTitle('Audio Transcriber Pro')
        self.setGeometry(420, 300, 800, 500)

        main_layout = QVBoxLayout(self)
        self._create_top_controls(main_layout)
        self._create_progress_bar(main_layout)
        self._create_text_panels(main_layout)

        self._update_ui_state()
        self.show()

    def _create_top_controls(self, parent_layout: QVBoxLayout) -> None:
        """Create recording controls and model selection"""
        top_layout = QHBoxLayout()

        # Recording Button
        self.recording_button = QPushButton('Start Recording', self)
        self._set_button_style('ready')
        self.recording_button.clicked.connect(self.toggle_recording)
        top_layout.addWidget(self.recording_button, 60)

        # Model Selection
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.available_models)
        self._configure_model_selector()
        top_layout.addWidget(self.model_selector, 30)

        # Re-Refine Button
        self.re_refine_button = QPushButton('Re-Refine Text', self)
        self.re_refine_button.setStyleSheet(self._refine_button_style())
        self.re_refine_button.clicked.connect(self.re_refine_text)
        top_layout.addWidget(self.re_refine_button, 10)

        parent_layout.addLayout(top_layout)

    def _configure_model_selector(self) -> None:
        """Initialize model dropdown with styling"""
        if 'phi4:latest' in self.available_models:
            self.model_selector.setCurrentText('phi4:latest')
        elif self.available_models:
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

    def _create_progress_bar(self, parent_layout: QVBoxLayout) -> None:
        """Initialize progress indicator"""
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        parent_layout.addWidget(self.progress_bar)

    def _create_text_panels(self, parent_layout: QVBoxLayout) -> None:
        """Create text input/output panels"""
        text_layout = QHBoxLayout()

        # Transcription Panel
        self.transcription_box = QTextEdit()
        self.transcription_box.setPlaceholderText("Original transcription...")
        text_layout.addLayout(self._create_text_panel(
            self.transcription_box, 'Copy', self.copy_transcription
        ))

        # Refined Text Panel
        self.refined_box = QTextEdit()
        self.refined_box.setPlaceholderText("Refined text...")
        text_layout.addLayout(self._create_text_panel(
            self.refined_box, 'Copy Refined', self.copy_refined
        ))

        parent_layout.addLayout(text_layout)

    def _create_text_panel(self, text_edit: QTextEdit,
                          btn_text: str, callback) -> QVBoxLayout:
        """Helper for creating text panel with copy button"""
        layout = QVBoxLayout()
        layout.addWidget(text_edit)

        btn = QPushButton(btn_text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)

        return layout

    def _set_button_style(self, state: str) -> None:
        """Update recording button appearance"""
        styles = {
            'ready': ('Start Recording', '#1E5631', '#2E8B57'),
            'recording': ('Stop Recording', '#8B0000', '#A52A2A'),
            'processing': ('Processing...', '#8B4500', '#CD6600')
        }

        text, bg_color, hover_color = styles.get(state, styles['ready'])
        self.recording_button.setText(text)
        self.recording_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {hover_color}; }}
        """)

    def _refine_button_style(self) -> str:
        """Style for secondary buttons"""
        return """
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #34495e; }
        """

    def _update_ui_state(self) -> None:
        """Enable/disable controls based on state"""
        enable = not self.state.has_active_threads
        self.recording_button.setEnabled(enable)
        self.re_refine_button.setEnabled(enable)
        self.model_selector.setEnabled(enable)

    # --------------------------
    # Core Functionality
    # --------------------------
    def toggle_recording(self) -> None:
        """Handle recording start/stop"""
        with self._processing_lock:
            if self.state.has_active_threads:
                return

            if not self.state.is_recording:
                self._start_recording()
            else:
                self._stop_recording()

    def _start_recording(self) -> None:
        """Initiate audio recording"""
        self.state.is_recording = True
        self._set_button_style('recording')
        self.progress_bar.setValue(0)

        threading.Thread(
            target=self._record_audio_background,
            daemon=True
        ).start()

    def _record_audio_background(self) -> None:
        """Background recording thread"""
        with AudioRecorder(self.state) as recorder:
            recorder.record_audio()

        if self.state.audio_buffer:
            self._process_audio()

    def _stop_recording(self) -> None:
        """Handle recording cessation"""
        self.state.is_recording = False
        self._set_button_style('processing')

    def _process_audio(self) -> None:
        """Handle audio processing pipeline"""
        worker = TranscriptionThread(
            self.state.audio_buffer,
            self.model_selector.currentText(),
            self.state
        )
        worker.transcription_finished.connect(self._handle_transcription)
        worker.refinement_finished.connect(self._handle_refinement)
        worker.finished.connect(self._update_ui_state)
        worker.start()

        # Clear audio buffer after processing
        self.state.audio_buffer = b''

    def re_refine_text(self) -> None:
        """Handle manual refinement requests"""
        text = self.transcription_box.toPlainText().strip()
        if not text:
            return

        self._set_button_style('processing')
        self.progress_bar.setValue(50)

        worker = RefinementThread(
            text,
            self.model_selector.currentText(),
            self.state
        )
        worker.refinement_finished.connect(self._handle_refinement)
        worker.finished.connect(self._update_ui_state)
        worker.start()

    def _handle_transcription(self, text: str) -> None:
        """Update UI with transcription results"""
        self.transcription_box.setText(text)
        self.progress_bar.setValue(50)

    def _handle_refinement(self, text: str) -> None:
        """Update UI with refinement results"""
        self.refined_box.setText(text)
        self.progress_bar.setValue(100)
        self._set_button_style('ready')

    def copy_transcription(self) -> None:
        """Copy transcription text to clipboard"""
        QApplication.clipboard().setText(self.transcription_box.toPlainText())

    def copy_refined(self) -> None:
        """Copy refined text to clipboard"""
        QApplication.clipboard().setText(self.refined_box.toPlainText())

    # --------------------------
    # Event Handling
    # --------------------------
    def closeEvent(self, event) -> None:
        """Ensure clean shutdown"""
        if self.state.has_active_threads:
            logger.info("Waiting for active processes to complete...")
            while self.state.has_active_threads:
                QApplication.processEvents()

        event.accept()

# --------------------------
# Application Entry Point
# --------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    sys.exit(app.exec_())