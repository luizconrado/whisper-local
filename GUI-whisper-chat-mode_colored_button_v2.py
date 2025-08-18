import sys
import threading
import datetime
import re
import pyaudio
import wave
import tempfile
import logging
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
format = pyaudio.paInt16
channels = 1
chunk_size = 1024

# GUI variables
is_recording = False
audio = None
stream = None
audio_bytes = None
recording_finished = threading.Event()


def record_audio():
    global audio, stream, is_recording, audio_bytes, recording_finished
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        recorded_data = []
        while is_recording:
            data = stream.read(chunk_size, exception_on_overflow=False)
            recorded_data.append(data)
        audio_bytes = b''.join(recorded_data)
    except Exception as e:
        logging.error(f"Recording error: {e}")
        audio_bytes = None
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        recording_finished.set()


def start_recording():
    global is_recording
    is_recording = True
    recording_finished.clear()
    threading.Thread(target=record_audio, daemon=True).start()


def stop_recording():
    global is_recording
    is_recording = False


class TranscriptionThread(QThread):
    transcription_finished = pyqtSignal(str)
    refinement_finished = pyqtSignal(str)

    def __init__(self, audio_data, model):
        super().__init__()
        self.audio_data = audio_data
        self.model = model

    def run(self):
        if self.audio_data:
            transcription_result = self.transcribe_audio()
            self.transcription_finished.emit(transcription_result)
            if "Failed to transcribe" not in transcription_result:
                self.refine_text(transcription_result)
            else:
                self.refinement_finished.emit("")
        else:
            self.transcription_finished.emit("No audio data to transcribe.")
            self.refinement_finished.emit("")

    def transcribe_audio(self):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
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
            return f"Failed to transcribe: {e}"

    def refine_text(self, text):
        try:
            response = ollama.chat(
                model=self.model,
                messages=self._create_messages(text),
                options={'ctx_num': 8000, 'temperature': 0.2, 'seed': 1}
            )
            refined = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL).strip().strip('"')
            self.refinement_finished.emit(refined)
        except Exception as e:
            self.refinement_finished.emit(f"Refinement failed: {e}")

    def _create_messages(self, text):
        return [
            {
                'role': 'system',
                'content': (
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
            },
            {
                'role': 'user',
                'content': f'Here is the text to be corrected: "{text}"',
            },
        ]


class RefinementThread(QThread):
    refinement_finished = pyqtSignal(str)

    def __init__(self, text, model):
        super().__init__()
        self.text = text
        self.model = model

    def run(self):
        try:
            response = ollama.chat(
                model=self.model,
                messages=self._create_messages(self.text),
                options={'ctx_num': 8000, 'temperature': 0.2, 'seed': 1}
            )
            refined = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL).strip().strip('"')
            self.refinement_finished.emit(refined)
        except Exception as e:
            self.refinement_finished.emit(f"Refinement failed: {e}")

    def _create_messages(self, text):
        return [
            {
                'role': 'system',
                'content': (
                    'You are my text corrector. You should never answer any questions. '
                    'Your task is only to correct any spelling discrepancies in the transcribed text, '
                    'improve my vocabulary when necessary, making the text clear and easy to understand. '
                    'Also, add punctuation such as periods, commas, and capitalization. '
                    'Please use only the context provided. As the output, I only want the corrected text, '
                    'no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.'
                ),
            },
            {
                'role': 'user',
                'content': f'Here is the text to be corrected: "{self.text}"',
            },
        ]


class AudioTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()
        self.available_models = self.fetch_models()
        self.current_transcription = ""
        self.initUI()

    def fetch_models(self):
        try:
            models = ollama.list()['models']
            installed_models = [m['name'] for m in models]

            if 'phi4:latest' in installed_models:
                installed_models.remove('phi4:latest')
                installed_models.insert(0, 'phi4:latest')

            return installed_models or ['phi4:latest', 'deepseek-r1:1.5b']
        except Exception as e:
            logging.error(f"Model fetch error: {e}")
            return ['phi4:latest', 'deepseek-r1:1.5b']

    def initUI(self):
        main_layout = QVBoxLayout(self)
        self._create_top_controls(main_layout)
        self._create_progress_bar(main_layout)
        self._create_text_areas(main_layout)
        self.setWindowTitle('Audio Transcriber')
        self.setGeometry(420, 300, 800, 500)
        self.show()

    def _create_top_controls(self, main_layout):
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
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

    def _create_text_areas(self, main_layout):
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
        global audio_bytes
        if not is_recording:
            self.set_button_style('recording')
            start_recording()
            self.progress_bar.setValue(0)
        else:
            self.set_button_style('transcribing')
            stop_recording()
            recording_finished.wait()
            if audio_bytes:
                self.current_transcription = ""
                self.worker = TranscriptionThread(audio_bytes, self.model_selector.currentText())
                self.worker.transcription_finished.connect(self.display_transcription)
                self.worker.refinement_finished.connect(self.display_refined_text)
                self.worker.start()

    def re_refine_text(self):
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
        self.transcription_box.setText(text)
        self.current_transcription = text
        self.progress_bar.setValue(50)

    def display_refined_text(self, text):
        self.text_refined_box.setText(text)
        self.progress_bar.setValue(100)
        self.set_button_style('ready')

    def copy_text(self, widget):
        clipboard = QApplication.clipboard()
        clipboard.setText(widget.toPlainText())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioTranscriberApp()
    sys.exit(app.exec_())