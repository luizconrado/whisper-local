import sys
import threading
import os
import datetime
import pyaudio
import wave
from pydub import AudioSegment
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import pyqtSignal, QThread, Qt
import mlx_whisper  # Ensure mlx_whisper is properly imported
import ollama

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
    global audio, stream, is_recording, recording_file_name, recording_finished

    audio = pyaudio.PyAudio()
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

    stream.stop_stream()
    stream.close()
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
    finished = pyqtSignal(str, str)  # Emit both original and refined text

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        if self.file_path:
            transcription_result = self.transcribe_audio(self.file_path)
            refined_text = self.refine_text(transcription_result)
            self.finished.emit(transcription_result, refined_text)
        else:
            self.finished.emit("No file to transcribe.", "")

    def transcribe_audio(self, file_path):
        print(f"Transcribing the audio file {file_path}...")
        try:
            result = mlx_whisper.transcribe(file_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
            return result['text']
        except Exception as e:
            error_message = f"Failed to transcribe: {e}"
            print(error_message)
            return error_message

    def refine_text(self, text):
        response = ollama.chat(model='llama3:8b-instruct-fp16', messages=[
            {
                'role': 'system',
                # 'content': 'You are my English corrector. Your task is to only correct any spelling discrepancies in the transcribed text, improve my English vocabulary when necessary, using a C1 level of English. Also, add punctuation such as periods, commas, and capitalization. Please use only the context provided. As the output, I only want the corrected text, no preamble, nothing else but the corrected text.',
                'content': 'You are my text corrector. You should never answer any questions. Your task is only to only correct any spelling discrepancies in the transcribed text, improve my vocabulary when necessary. Also, add punctuation such as periods, commas, and capitalization. Please use only the context provided. As the output, I only want the corrected text, no preamble, nothing else but the corrected text.',
            },
            {
                'role': 'user',
                # 'content': text,
                'content': 'Here is the text to be corrected: "' + text + '"',
            },
        ])
        return response['message']['content']


# PyQt5 Application
class AudioTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Recording button at the top
        self.recording_button = QPushButton('Start Recording', self)
        self.recording_button.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.recording_button)

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
        self.setGeometry(300, 300, 800, 400)
        self.show()

    def toggle_recording(self):
        global recording_file_name
        if not is_recording:
            self.recording_button.setText('Stop Recording')
            start_recording()
        else:
            self.recording_button.setText('Start Recording')
            stop_recording()
            recording_finished.wait()
            self.worker = TranscriptionThread(recording_file_name)
            self.worker.finished.connect(self.display_transcription)
            self.worker.start()

    def display_transcription(self, original_text, refined_text):
        self.transcription_box.setText(original_text)
        self.text_refined_box.setText(refined_text)

    def copy_text(self, text_edit):
        clipboard = QApplication.clipboard()
        clipboard.setText(text_edit.toPlainText())


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioTranscriberApp()
    sys.exit(app.exec_())
