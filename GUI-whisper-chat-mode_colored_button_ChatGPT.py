import sys
import threading
import datetime
import re
import pyaudio
import wave
import tempfile
import logging
from contextlib import closing
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QComboBox
)
from PyQt5.QtCore import pyqtSignal, QThread
import mlx_whisper
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio configuration
sample_rate = 44100
audio_format = pyaudio.paInt16  # renamed from "format" to avoid shadowing built-in names
channels = 1
chunk_size = 1024

# Define per-model defaults for the system message, user prompt, and chat options.
MODEL_DEFAULTS = {
    'phi4:latest': {
        'system_message': (
            'You are my text corrector. You should never answer any questions. '
            'Your task is only to correct any spelling discrepancies in the transcribed text, '
            'improve my vocabulary when necessary, making the text clear and easy to understand. '
            'Also, add punctuation such as periods, commas, and capitalization. '
            'Please use only the context provided. As the output, I only want the corrected text, '
            'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
        ),
        'user_message': 'Here is the text to be corrected:',
        'options': {'ctx_num': 8000, 'temperature': 0.2, 'seed': 1},
    },
    'deepseek-r1:1.5b': {
        'system_message': (
            'You are my text corrector (deepseek variant). '
            'Focus on correcting spelling discrepancies and punctuation. '
            'Improve vocabulary only if necessary, and return only the corrected text without extra commentary.'
        ),
        'user_message': 'Please review the following transcription for corrections:',
        'options': {'ctx_num': 6000, 'temperature': 0.3, 'seed': 2},
    },
    # You can add more models and their defaults here.
}


def create_refinement_payload(model: str, text: str):
    """
    Returns the messages payload and options for a given model and input text.

    This function builds the payload that will be sent to the ollama.chat API based on the model's
    defaults. It uses the system message, user message (introduction), and chat options specified in the
    MODEL_DEFAULTS configuration.
    """
    defaults = MODEL_DEFAULTS.get(model, {
        'system_message': 'Default system message: Please correct the text below.',
        'user_message': 'Here is the text to be corrected:',
        'options': {'ctx_num': 8000, 'temperature': 0.2, 'seed': 1},
    })
    messages = [
        {
            'role': 'system',
            'content': defaults['system_message'],
        },
        {
            'role': 'user',
            'content': f'{defaults["user_message"]} "{text}"',
        },
    ]
    return messages, defaults['options']


def refine_text_with_model(model: str, text: str) -> str:
    """
    Calls ollama.chat using the payload built from model-specific defaults and returns the refined text.

    This function uses create_refinement_payload to build the message list and options for the given model,
    then calls ollama.chat to get the refined text. Any <think> tags in the response are removed.
    """
    try:
        messages, options = create_refinement_payload(model, text)
        response = ollama.chat(
            model=model,
            messages=messages,
            options=options
        )
        refined = re.sub(
            r'<think>.*?</think>',
            '',
            response['message']['content'],
            flags=re.DOTALL
        ).strip().strip('"')
        return refined
    except Exception as e:
        logging.error("Refinement failed", exc_info=True)
        return f"Refinement failed: {e}"


class AudioRecorderThread(QThread):
    """
    AudioRecorderThread is a QThread that handles recording audio from the microphone.

    It uses PyAudio to capture audio data in chunks, concatenates the data, and emits a signal
    with the recorded audio bytes once recording is stopped. This class encapsulates all the audio recording
    functionality to avoid using global variables and to ensure proper resource management.
    """
    recording_finished = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_recording = False

    def run(self):
        """
        Records audio from the microphone using PyAudio.

        This method opens an audio stream with proper error handling and context management.
        It reads audio data in chunks until recording is stopped, then emits the concatenated audio bytes.
        """
        recorded_data = []
        try:
            with closing(pyaudio.PyAudio()) as audio:
                stream = audio.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size
                )
                while self._is_recording:
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        recorded_data.append(data)
                    except Exception as e:
                        logging.error("Error while reading audio stream", exc_info=True)
                        break
                stream.stop_stream()
                stream.close()
                recorded = b''.join(recorded_data)
        except Exception as e:
            logging.error("Recording error", exc_info=True)
            recorded = b''
        self.recording_finished.emit(recorded)

    def start_recording(self):
        """
        Starts the audio recording by setting the recording flag and starting the thread.
        """
        self._is_recording = True
        self.start()

    def stop_recording(self):
        """
        Stops the audio recording by clearing the recording flag.
        """
        self._is_recording = False


class TranscriptionThread(QThread):
    """
    TranscriptionThread is a QThread that handles the transcription and subsequent refinement of audio data.

    It uses mlx_whisper to transcribe the audio and then calls ollama.chat (via the refine_text_with_model helper)
    to refine the transcription by correcting spelling, punctuation, and other text issues.
    """
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)

    def __init__(self, audio_data, model):
        super().__init__()
        self.audio_data = audio_data
        self.model = model

    def run(self):
        """
        The run method first transcribes the audio data using mlx_whisper. If transcription is successful,
        it then refines the transcription using the refine_text_with_model helper. Signals are emitted for
        both transcription and refinement completion.
        """
        if self.audio_data:
            transcription_result = self.transcribe_audio()
            self.transcription_finished.emit(transcription_result)
            if "Failed to transcribe" not in transcription_result:
                refined = refine_text_with_model(self.model, transcription_result)
                self.refinement_finished.emit(refined)
            else:
                self.refinement_finished.emit("")
        else:
            self.transcription_finished.emit("No audio data to transcribe.")
            self.refinement_finished.emit("")

    def transcribe_audio(self):
        """
        Transcribes the recorded audio using mlx_whisper.

        This method creates a temporary WAV file from the recorded audio data and calls mlx_whisper.transcribe.
        It returns the transcribed text or an error message if transcription fails.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
                    wf.setframerate(sample_rate)
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
            logging.error("Transcription failed", exc_info=True)
            return f"Failed to transcribe: {e}"


class RefinementThread(QThread):
    """
    RefinementThread is a QThread that handles the refinement of already transcribed text.

    It uses the refine_text_with_model helper to call ollama.chat with model-specific configurations and returns
    the refined text.
    """
    refinement_finished = pyqtSignal(str)

    def __init__(self, text, model):
        super().__init__()
        self.text = text
        self.model = model

    def run(self):
        """
        The run method calls refine_text_with_model to refine the provided text and emits a signal when finished.
        """
        refined = refine_text_with_model(self.model, self.text)
        self.refinement_finished.emit(refined)


class AudioTranscriberApp(QWidget):
    """
    AudioTranscriberApp is the main PyQt5 application window.

    It provides an interface for recording audio, transcribing it using mlx_whisper, refining the transcription
    using ollama.chat, and displaying both the original transcription and the refined text. The user can also
    select the language model from a drop-down list and re-refine the text as needed.
    """

    def __init__(self):
        super().__init__()
        self.available_models = self.fetch_models()
        self.current_transcription = ""
        self.audio_recorder = None  # Will hold an instance of AudioRecorderThread when recording
        self.initUI()

    def fetch_models(self):
        """
        Fetches the list of available language models using ollama.list.

        If available, it prioritizes 'phi4:latest' and returns a list of model names. If fetching fails, it defaults
        to a pre-defined list of models.
        """
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]

            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')

            return installed_models or ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logging.error(f"Model fetch error: {e}", exc_info=True)
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def initUI(self):
        """
        Initializes the user interface, including top controls, progress bar, and text areas for transcription and
        refined text.
        """
        main_layout = QVBoxLayout(self)
        self._create_top_controls(main_layout)
        self._create_progress_bar(main_layout)
        self._create_text_areas(main_layout)
        self.setWindowTitle('Audio Transcriber')
        self.setGeometry(420, 300, 800, 500)
        self.show()

    def _create_top_controls(self, main_layout):
        """
        Creates the top control panel which includes the recording button, model selection drop-down, and re-refine button.
        """
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

    def _create_progress_bar(self, main_layout):
        """
        Creates a progress bar to indicate the status of recording, transcription, and refinement.
        """
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

    def _create_text_areas(self, main_layout):
        """
        Creates two text areas: one for the original transcription and one for the refined text. Also adds copy buttons.
        """
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

    def set_button_style(self, state):
        """
        Sets the style of the recording button based on the current state (ready, recording, or transcribing).
        """
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

    def toggle_recording(self):
        """
        Toggles recording on and off. Starts recording if not currently recording; stops and processes audio otherwise.
        """
        # If not currently recording, start recording.
        if self.audio_recorder is None:
            self.set_button_style('recording')
            self.progress_bar.setValue(0)
            self.audio_recorder = AudioRecorderThread()
            self.audio_recorder.recording_finished.connect(self.on_recording_finished)
            self.audio_recorder.start_recording()
        else:
            # If recording, stop recording. The recording thread will emit a signal when done.
            self.set_button_style('transcribing')
            self.audio_recorder.stop_recording()

    def on_recording_finished(self, recorded_bytes):
        """
        Slot called when audio recording is finished. Initiates transcription and refinement if audio was recorded.
        """
        self.audio_recorder = None  # Reset the recorder instance
        if recorded_bytes:
            self.current_transcription = ""
            # Start the transcription thread with the recorded audio and selected model.
            self.worker = TranscriptionThread(recorded_bytes, self.model_selector.currentText())
            self.worker.transcription_finished.connect(self.display_transcription)
            self.worker.refinement_finished.connect(self.display_refined_text)
            self.worker.start()
        else:
            self.display_transcription("No audio data to transcribe.")
            self.display_refined_text("")

    def re_refine_text(self):
        """
        Re-refines the transcription text using the currently selected model.
        """
        if not self.transcription_box.toPlainText().strip():
            return

        self.set_button_style('transcribing')
        self.progress_bar.setValue(50)
        self.current_transcription = self.transcription_box.toPlainText()

        self.refinement_worker = RefinementThread(
            self.current_transcription,
            self.model_selector.currentText()
        )
        self.refinement_worker.refinement_finished.connect(self.display_refined_text)
        self.refinement_worker.start()

    def display_transcription(self, text):
        """
        Displays the transcription in the transcription text area and updates the progress bar.
        """
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

    def display_refined_text(self, text):
        """
        Displays the refined text in the refined text area, updates the progress bar, and resets the button style.
        """
        self.text_refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style('ready')

    def copy_text(self, widget):
        """
        Copies the content of the specified text widget to the clipboard.
        """
        clipboard = QApplication.clipboard()
        clipboard.setText(widget.toPlainText())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioTranscriberApp()
    sys.exit(app.exec_())
