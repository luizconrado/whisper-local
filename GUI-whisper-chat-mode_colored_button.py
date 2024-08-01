import sys
import threading
import os
import datetime
import pyaudio
import wave
from pydub import AudioSegment
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QProgressBar
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QColor
import mlx_whisper
import ollama
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio configuration
sample_rate = 44100
format = pyaudio.paInt16  # 16-bit audio format
channels = 1
timestamp_format = '%Y-%m-%d_%H-%M-%S'

# GUI variables
is_recording = False
audio = None
stream = None
recording_file_name = None
recording_finished = threading.Event()  # Threading event to indicate recording completion


def record_audio():
    """
    Record audio and save it as a temporary WAV file and a final MP3 file.
    """
    global audio, stream, is_recording, recording_file_name, recording_finished

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=format,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=1024)

        recorded_data = []
        while is_recording:
            data = stream.read(1024, exception_on_overflow=False)
            recorded_data.append(data)

        temp_wav_file = f"temp_{datetime.datetime.now().strftime(timestamp_format)}.wav"
        with wave.open(temp_wav_file, 'wb') as output_file:
            output_file.setnchannels(channels)
            output_file.setsampwidth(audio.get_sample_size(format))
            output_file.setframerate(sample_rate)
            output_file.writeframes(b''.join(recorded_data))

        recording_file_name = f"recorded_audio_{datetime.datetime.now().strftime(timestamp_format)}.mp3"
        sound = AudioSegment.from_wav(temp_wav_file)
        sound.export(recording_file_name, format="mp3")

        os.remove(temp_wav_file)
    except Exception as e:
        logging.error(f"Error during recording: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if audio is not None:
            audio.terminate()
        recording_finished.set()


def start_recording():
    """
    Start the audio recording in a separate thread.
    """
    global is_recording
    is_recording = True
    recording_finished.clear()
    threading.Thread(target=record_audio, daemon=True).start()


def stop_recording():
    """
    Stop the audio recording.
    """
    global is_recording
    is_recording = False


class TranscriptionThread(QThread):
    transcription_finished = pyqtSignal(str)  # Emit the original text
    refinement_finished = pyqtSignal(str)  # Emit the refined text

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """
        Run the transcription and refinement process.
        """
        if self.file_path:
            transcription_result = self.transcribe_audio(self.file_path)
            self.transcription_finished.emit(transcription_result)
            if "Failed to transcribe" not in transcription_result:
                refined_text = self.refine_text(transcription_result)
                self.refinement_finished.emit(refined_text)
            else:
                self.refinement_finished.emit("")
            try:
                os.remove(self.file_path)
            except OSError as e:
                logging.error(f"Error deleting file: {e}")
        else:
            self.transcription_finished.emit("No file to transcribe.")
            self.refinement_finished.emit("")

    def transcribe_audio(self, file_path):
        """
        Transcribe the audio file using mlx_whisper.
        """
        logging.info(f"Transcribing the audio file {file_path}...")
        try:
            result = mlx_whisper.transcribe(file_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
            return result['text'].strip()
        except Exception as e:
            error_message = f"Failed to transcribe: {e}"
            logging.error(error_message)
            return error_message

    def refine_text(self, text):
        """
        Refine the transcribed text using ollama.
        Best models at the moment: phi3:14b & llama3.1:latest
        """
        response = ollama.chat(model='phi3:14b', messages=[
            {
                'role': 'system',
                'content': 'You are my text corrector. You should never answer any questions. Your task is only to only correct any spelling discrepancies in the transcribed text, improve my vocabulary when necessary, making the text clear and easy to understand. Also, add punctuation such as periods, commas, and capitalization. Please use only the context provided. As the output, I only want the corrected text, no preamble, introduction, notes, or explanations. Only the corrected text and nothing else.',
            },
            {
                'role': 'user',
                'content': 'Here is the text to be corrected: "' + text + '"',
            },
        ],
                               options={
                                   'ctx_num': 8000,
                                   'temperature': 0.2,
                                   'seed': 1
                               })
        return response['message']['content']


class AudioTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface.
        """
        main_layout = QVBoxLayout(self)

        # Recording button (full width, reduced height)
        self.recording_button = QPushButton('Start Recording', self)
        self.set_button_style('ready')
        self.recording_button.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.recording_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Text boxes and copy buttons
        text_layout = QHBoxLayout()

        # Transcription box and copy button
        vbox1 = QVBoxLayout()
        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Transcription will appear here...")
        vbox1.addWidget(self.transcription_box)
        self.copy_transcription_button = QPushButton('Copy Text', self)
        self.copy_transcription_button.clicked.connect(
            lambda: self.copy_text(self.transcription_box)
        )
        vbox1.addWidget(self.copy_transcription_button)
        text_layout.addLayout(vbox1)

        # Refined text box and copy button
        vbox2 = QVBoxLayout()
        self.text_refined_box = QTextEdit(self)
        self.text_refined_box.setPlaceholderText("Refined text will appear here...")
        vbox2.addWidget(self.text_refined_box)
        self.copy_refined_button = QPushButton('Copy Text', self)
        self.copy_refined_button.clicked.connect(
            lambda: self.copy_text(self.text_refined_box)
        )
        vbox2.addWidget(self.copy_refined_button)
        text_layout.addLayout(vbox2)

        main_layout.addLayout(text_layout)

        self.setWindowTitle('Audio Transcriber')
        self.setGeometry(420, 300, 800, 500)
        self.setLayout(main_layout)
        self.show()

    def set_button_style(self, state):
        """
        Set the style of the recording button based on its state.
        """
        styles = {
            'ready': {
                'text': 'Start Recording',
                'bg_color': '#1E5631',  # Darker green
                'hover_color': '#2E8B57'  # Slightly lighter green for hover
            },
            'recording': {
                'text': 'Stop Recording',
                'bg_color': '#8B0000',  # Dark red
                'hover_color': '#A52A2A'  # Slightly lighter red for hover
            },
            'transcribing': {
                'text': 'Transcribing...',
                'bg_color': '#8B4500',  # Dark orange
                'hover_color': '#CD6600'  # Slightly lighter orange for hover
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
        Toggle the recording state and start/stop the recording process.
        """
        global recording_file_name
        if not is_recording:
            self.set_button_style('recording')
            start_recording()
            self.progress_bar.setValue(0)
        else:
            self.set_button_style('transcribing')
            stop_recording()
            recording_finished.wait()
            self.worker = TranscriptionThread(recording_file_name)
            self.worker.transcription_finished.connect(self.display_transcription)
            self.worker.refinement_finished.connect(self.display_refined_text)
            self.worker.start()

    def display_transcription(self, original_text):
        """
        Display the transcribed text in the transcription box.
        """
        self.transcription_box.setText(original_text)
        self.progress_bar.setValue(50)

    def display_refined_text(self, refined_text):
        """
        Display the refined text in the refined text box.
        """
        self.text_refined_box.setText(refined_text)
        self.progress_bar.setValue(100)
        self.set_button_style('ready')

    def copy_text(self, text_edit):
        """
        Copy text from the text edit widget to the clipboard.
        """
        clipboard = QApplication.clipboard()
        clipboard.setText(text_edit.toPlainText())


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioTranscriberApp()
    sys.exit(app.exec_())