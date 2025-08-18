"""
Audio Transcriber Application

This application provides audio transcription and text refinement capabilities using
various language models. It includes a GUI interface for recording audio, transcribing
it, and refining the transcribed text using different LLM models.
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import sys
import threading
import datetime
import re
import pyaudio
import wave
import tempfile
import logging
import queue
from functools import partial
from contextlib import contextmanager
from enum import Enum
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox,
    QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
import mlx_whisper
import ollama

# Setup logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_transcriber.log')
    ]
)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Enum for different types of system messages."""
    TEXT_CORRECTION = "text_correction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    # Add more message types as needed


@dataclass
class MessageTemplate:
    """Template for different types of messages."""
    system_message: str
    user_message_template: str  # Will use Python string formatting


@dataclass
class ModelConfig:
    """Configuration class for LLM models."""
    name: str
    ctx_num: int
    temperature: float
    seed: Optional[int] = None
    message_templates: Dict[MessageType, MessageTemplate] = field(default_factory=dict)

    @classmethod
    def get_default_configs(cls) -> Dict[str, 'ModelConfig']:
        """Returns default configurations for supported models."""
        return {
            'phi4:latest': cls(
                name='phi4:latest',
                ctx_num=8000,
                temperature=0.2,
                seed=1,
                message_templates={
                    MessageType.TEXT_CORRECTION: MessageTemplate(
                        system_message=(
                            'You are my text corrector. You should never answer any questions. '
                            'Your task is only to correct any spelling discrepancies in the transcribed text, '
                            'improve my vocabulary when necessary, making the text clear and easy to understand. '
                            'Also, add punctuation such as periods, commas, and capitalization. '
                            'Please use only the context provided. As the output, I only want the corrected text, '
                            'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                        ),
                        user_message_template='Here is the text to be corrected: "{text}"'
                    ),
                    MessageType.SUMMARIZATION: MessageTemplate(
                        system_message=(
                            'You are a concise summarizer. Create clear, accurate summaries '
                            'that capture the main points while maintaining the original meaning. '
                            'Focus on key information and remove redundancy.'
                        ),
                        user_message_template='Please summarize the following text: "{text}"'
                    )
                }
            ),
            'deepseek-r1:1.5b': cls(
                name='deepseek-r1:1.5b',
                ctx_num=4000,
                temperature=0.1,
                seed=1,
                message_templates={
                    MessageType.TEXT_CORRECTION: MessageTemplate(
                        system_message=(
                            'You are a precise text corrector. Focus on maintaining the original meaning while '
                            'fixing spelling, grammar, and punctuation. Improve clarity without changing the '
                            'core message. Return only the corrected text without any additional commentary.'
                        ),
                        user_message_template='Correct this transcribed text: "{text}"'
                    ),
                    MessageType.SUMMARIZATION: MessageTemplate(
                        system_message=(
                            'Create brief, accurate summaries focusing on essential information. '
                            'Maintain clarity and precision while being concise.'
                        ),
                        user_message_template='Generate a summary of: "{text}"'
                    )
                }
            )
        }

    @classmethod
    def get_config(cls, model_name: str) -> 'ModelConfig':
        """
        Get configuration for a specific model, falling back to defaults if not found.

        Args:
            model_name: Name of the model to get configuration for

        Returns:
            ModelConfig: Configuration for the specified model or default configuration
        """
        configs = cls.get_default_configs()
        if model_name not in configs:
            base_config = configs['phi4:latest']
            return cls(
                name=model_name,
                ctx_num=base_config.ctx_num,
                temperature=base_config.temperature,
                seed=base_config.seed,
                message_templates=base_config.message_templates
            )
        return configs[model_name]


class AudioConfig:
    """Configuration class for audio settings."""
    SAMPLE_RATE: int = 44100
    FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    MIN_AUDIO_LENGTH: float = 0.5  # Minimum audio length in seconds
    MAX_AUDIO_LENGTH: float = 300.0  # Maximum audio length in seconds

    @classmethod
    def validate_audio_length(cls, audio_length: float) -> bool:
        """Validate audio length against configured limits."""
        return cls.MIN_AUDIO_LENGTH <= audio_length <= cls.MAX_AUDIO_LENGTH


class AudioManager:
    """Manages audio recording resources and state."""

    def __init__(self) -> None:
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._is_recording: bool = False
        self._recording_finished: threading.Event = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        """Thread-safe access to recording state."""
        with self._lock:
            return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool) -> None:
        """Thread-safe modification of recording state."""
        with self._lock:
            self._is_recording = value

    @contextmanager
    def audio_session(self):
        """Context manager for audio resources."""
        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                frames_per_buffer=AudioConfig.CHUNK_SIZE
            )
            yield
        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._audio:
                self._audio.terminate()
            self._recording_finished.set()

    def record_audio(self) -> None:
        """Record audio data and manage resources."""
        try:
            with self.audio_session():
                recorded_data = []
                start_time = datetime.datetime.now()

                while self.is_recording:
                    if self._stream is None:
                        raise RuntimeError("Audio stream is not initialized")

                    data = self._stream.read(AudioConfig.CHUNK_SIZE, exception_on_overflow=False)
                    recorded_data.append(data)

                    # Check recording duration
                    duration = (datetime.datetime.now() - start_time).total_seconds()
                    if duration > AudioConfig.MAX_AUDIO_LENGTH:
                        logger.warning("Maximum recording duration reached")
                        break

                audio_data = b''.join(recorded_data)
                audio_length = len(audio_data) / (
                        AudioConfig.SAMPLE_RATE * AudioConfig.CHANNELS *
                        pyaudio.get_sample_size(AudioConfig.FORMAT)
                )

                if AudioConfig.validate_audio_length(audio_length):
                    self._audio_queue.put(audio_data)
                else:
                    logger.error(f"Invalid audio length: {audio_length} seconds")
                    self._audio_queue.put(None)

        except Exception as e:
            logger.error(f"Recording error: {e}")
            self._audio_queue.put(None)

    def start_recording(self) -> None:
        """Start audio recording in a new thread."""
        self.is_recording = True
        self._recording_finished.clear()
        threading.Thread(target=self.record_audio, daemon=True).start()

    def stop_recording(self) -> Optional[bytes]:
        """
        Stop recording and return the audio data.

        Returns:
            Optional[bytes]: The recorded audio data or None if recording failed
        """
        self.is_recording = False
        self._recording_finished.wait()
        try:
            return self._audio_queue.get(timeout=1.0)
        except queue.Empty:
            logger.error("Timeout waiting for audio data")
            return None


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class TextRefinementService:
    """Service class for handling text refinement operations."""

    TIMEOUT_SECONDS: int = 30

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_config = ModelConfig.get_config(model_name)

    def refine_text(
            self,
            text: str,
            message_type: MessageType = MessageType.TEXT_CORRECTION
    ) -> str:
        """
        Refine text using the configured model.

        Args:
            text: The input text to refine
            message_type: The type of refinement to perform

        Returns:
            str: The refined text

        Raises:
            LLMError: If refinement fails or times out
        """
        try:
            # Create a timer for timeout
            timer = QTimer()
            timer.setSingleShot(True)
            timer.start(self.TIMEOUT_SECONDS * 1000)

            response = ollama.chat(
                model=self.model_name,
                messages=self._create_messages(text, message_type),
                options=self._get_chat_options()
            )

            if timer.isActive():
                timer.stop()
                return self._clean_response(response['message']['content'])
            else:
                raise LLMError("LLM request timed out")

        except Exception as e:
            raise LLMError(f"Refinement failed: {str(e)}")

    def _create_messages(
            self,
            text: str,
            message_type: MessageType
    ) -> List[Dict[str, str]]:
        """Create messages for the LLM based on message type."""
        template = self.model_config.message_templates.get(
            message_type,
            self.model_config.message_templates[MessageType.TEXT_CORRECTION]
        )

        return [
            {
                'role': 'system',
                'content': template.system_message,
            },
            {
                'role': 'user',
                'content': template.user_message_template.format(text=text),
            },
        ]

    def _get_chat_options(self) -> Dict[str, Any]:
        """Get chat options based on model configuration."""
        return {
            'ctx_num': self.model_config.ctx_num,
            'temperature': self.model_config.temperature,
            'seed': self.model_config.seed
        }

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean up the response text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip().strip('"')


class WorkerThread(QThread):
    """Base class for worker threads."""

    def __init__(self) -> None:
        super().__init__()
        self._is_running = True

    def stop(self) -> None:
        """Safely stop the thread."""
        self._is_running = False
        self.wait()


class TranscriptionThread(WorkerThread):
    """Thread for handling transcription and refinement."""

    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, audio_data: bytes, model: str) -> None:
        super().__init__()
        self.audio_data = audio_data
        self.refinement_service = TextRefinementService(model)

    def run(self) -> None:
        if not self.audio_data:
            self.error_occurred.emit("No audio data to transcribe.")
            return

        try:
            transcription_result = self._transcribe_audio()
            self.transcription_finished.emit(transcription_result)

            if "Failed to transcribe" not in transcription_result:
                refined_text = self.refinement_service.refine_text(transcription_result)
                self.refinement_finished.emit(refined_text)
            else:
                self.error_occurred.emit("Transcription failed")

        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def _transcribe_audio(self) -> str:
        """
        Transcribe audio data to text.

        Returns:
            str: The transcribed text

        Raises:
            Exception: If transcription fails
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(AudioConfig.CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(AudioConfig.FORMAT))
                    wf.setframerate(AudioConfig.SAMPLE_RATE)
                    wf.writeframes(self.audio_data)

                result = mlx_whisper.transcribe(
                    temp_wav.name,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    initial_prompt=(
                        "Accurate transcript with proper punctuation. "
                        "Do not repeat phrases. Use standard spelling."
                    )
                )
                return result['text'].strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise


class RefinementThread(WorkerThread):
    """Thread for handling text refinement operations."""

    refinement_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, text: str, model: str) -> None:
        super().__init__()
        self.text = text
        self.refinement_service = TextRefinementService(model)

    def run(self) -> None:
        try:
            refined_text = self.refinement_service.refine_text(self.text)
            self.refinement_finished.emit(refined_text)
        except LLMError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {str(e)}")


class AudioTranscriberApp(QWidget):
    """Main application window for audio transcription."""

    def __init__(self) -> None:
        super().__init__()
        self.audio_manager = AudioManager()
        self.available_models = self._fetch_models()
        self.current_transcription = ""
        self.current_worker: Optional[WorkerThread] = None
        self._init_ui()

    def closeEvent(self, event) -> None:
        """Handle application closure."""
        try:
            # Stop any ongoing recording
            if self.audio_manager.is_recording:
                self.audio_manager.stop_recording()

            # Clean up current worker thread
            if self.current_worker and self.current_worker.isRunning():
                self.current_worker.stop()

            event.accept()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            event.accept()

    def _fetch_models(self) -> List[str]:
        """Fetch available models from Ollama."""
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]

            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')

            return installed_models or ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logger.error(f"Model fetch error: {e}")
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        self._create_top_controls(main_layout)
        self._create_progress_bar(main_layout)
        self._create_text_areas(main_layout)

        self.setWindowTitle('Audio Transcriber')
        self.setGeometry(420, 300, 800, 500)
        self.show()

    def _create_top_controls(self, main_layout: QVBoxLayout) -> None:
        """Create the top control panel."""
        top_layout = QHBoxLayout()

        # Recording button
        self.recording_button = QPushButton('Start Recording', self)
        self.set_button_style('ready')
        self.recording_button.clicked.connect(self.toggle_recording)
        top_layout.addWidget(self.recording_button, 50)

        # Model selection
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.available_models)

        if 'phi4:latest' in self.available_models:
            self.model_selector.setCurrentText('phi4:latest')
        elif len(self.available_models) > 0:
            self.model_selector.setCurrentIndex(0)

        self.model_selector.setStyleSheet("""
            QComboBox { 
                padding: 8px; 
                min-width: 120px; 
                font-size: 14px; 
                border: 1px solid #cccccc; 
                border-radius: 4px; 
            }
            QComboBox::drop-down {
                width: 20px;
            }
        """)
        top_layout.addWidget(self.model_selector, 30)

        # Re-refine button
        self.re_refine_button = QPushButton('Re-Refine Text', self)
        self.re_refine_button.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { 
                background-color: #34495e; 
            }
        """)
        self.re_refine_button.clicked.connect(self.re_refine_text)
        top_layout.addWidget(self.re_refine_button, 20)

        main_layout.addLayout(top_layout)

    def _create_progress_bar(self, main_layout: QVBoxLayout) -> None:
        """Create the progress bar."""
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

    def _create_text_areas(self, main_layout: QVBoxLayout) -> None:
        """Create the text areas for transcription and refined text."""
        text_layout = QHBoxLayout()

        # Transcription panel
        vbox1 = QVBoxLayout()
        self.transcription_box = QTextEdit()
        self.transcription_box.setPlaceholderText("Original transcription...")
        vbox1.addWidget(self.transcription_box)

        self.copy_transcription_btn = QPushButton('Copy')
        self.copy_transcription_btn.clicked.connect(
            partial(self.copy_text, self.transcription_box)
        )
        vbox1.addWidget(self.copy_transcription_btn)
        text_layout.addLayout(vbox1)

        # Refined text panel
        vbox2 = QVBoxLayout()
        self.text_refined_box = QTextEdit()
        self.text_refined_box.setPlaceholderText("Refined text...")
        vbox2.addWidget(self.text_refined_box)

        self.copy_refined_btn = QPushButton('Copy')
        self.copy_refined_btn.clicked.connect(
            partial(self.copy_text, self.text_refined_box)
        )
        vbox2.addWidget(self.copy_refined_btn)
        text_layout.addLayout(vbox2)

        main_layout.addLayout(text_layout)

    def set_button_style(self, state: str) -> None:
        """Set the style of the recording button based on its state."""
        styles = {
            'ready': {
                'text': 'Start Recording',
                'bg_color': '#1E5631',
                'hover_color': '#2E8B57'
            },
            'recording': {
                'text': 'Stop Recording',
                'bg_color': '#8B0000',
                'hover_color': '#A52A2A'
            },
            'transcribing': {
                'text': 'Processing...',
                'bg_color': '#8B4500',
                'hover_color': '#CD6600'
            }
        }

        style = styles[state]
        self.recording_button.setText(style['text'])
        self.recording_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {style['bg_color']};
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {style['hover_color']};
            }}
        """)

    def _handle_error(self, error_message: str) -> None:
        """Handle and display errors."""
        logger.error(error_message)
        QMessageBox.critical(self, "Error", error_message)
        self.set_button_style('ready')
        self.progress_bar.setValue(0)

    def toggle_recording(self) -> None:
        """Toggle audio recording state."""
        if not self.audio_manager.is_recording:
            self.set_button_style('recording')
            self.audio_manager.start_recording()
            self.progress_bar.setValue(0)
        else:
            self.set_button_style('transcribing')
            audio_data = self.audio_manager.stop_recording()

            if audio_data:
                self.current_transcription = ""
                self.current_worker = TranscriptionThread(
                    audio_data,
                    self.model_selector.currentText()
                )
                self.current_worker.transcription_finished.connect(self.display_transcription)
                self.current_worker.refinement_finished.connect(self.display_refined_text)
                self.current_worker.error_occurred.connect(self._handle_error)
                self.current_worker.start()
            else:
                self._handle_error("Failed to capture audio")

    def re_refine_text(self) -> None:
        """Re-refine the current transcription."""
        text = self.transcription_box.toPlainText().strip()
        if not text:
            return

        self.set_button_style('transcribing')
        self.progress_bar.setValue(50)
        self.current_transcription = text

        self.current_worker = RefinementThread(
            text,
            self.model_selector.currentText()
        )
        self.current_worker.refinement_finished.connect(self.display_refined_text)
        self.current_worker.error_occurred.connect(self._handle_error)
        self.current_worker.start()

    def display_transcription(self, text: str) -> None:
        """Display the transcribed text."""
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

    def display_refined_text(self, text: str) -> None:
        """Display the refined text."""
        self.text_refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style('ready')

    def copy_text(self, widget: QTextEdit) -> None:
        """Copy text from the specified widget to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(widget.toPlainText())


def main() -> None:
    """Main application entry point with error handling."""
    try:
        app = QApplication(sys.argv)

        # Set application-wide exception handler
        sys.excepthook = lambda type, value, traceback: logger.critical(
            "Uncaught exception",
            exc_info=(type, value, traceback)
        )

        ex = AudioTranscriberApp()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()