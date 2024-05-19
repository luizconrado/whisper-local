import sys
import threading
import os
import datetime
import pyaudio
import wave
from pydub import AudioSegment
import mlx_whisper

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

    # Save the recorded audio to a temporary WAV file
    temp_wav_file = f"temp_{datetime.datetime.now().strftime(timestamp_format)}.wav"
    with wave.open(temp_wav_file, 'wb') as output_file:
        output_file.setnchannels(channels)
        output_file.setsampwidth(audio.get_sample_size(format))
        output_file.setframerate(sample_rate)
        output_file.writeframes(b''.join(recorded_data))

    # Convert WAV to MP3
    sound = AudioSegment.from_wav(temp_wav_file)
    sound.export(recording_file_name, format="mp3")

    # Delete the temporary WAV file
    os.remove(temp_wav_file)

    # Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Set the recording finished event
    recording_finished.set()

def start_recording():
    global is_recording, recording_file_name

    if not is_recording:
        # Define the recording file name
        recording_file_name = f"recorded_audio_{datetime.datetime.now().strftime(timestamp_format)}.mp3"

        is_recording = True
        threading.Thread(target=record_audio, daemon=True).start()

def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False



def transcribe_audio(file_path):
    """ Transcribes the given audio file to text using mlx-whisper. """
    print(f"Transcribing the audio file {file_path}...")
    try:
        result = mlx_whisper.transcribe(file_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
        print("Transcription:", result['text'])
    except Exception as e:
        print("Failed to transcribe:", e)


def main():
    print("Press 'r' to start/stop recording. Press 'q' and Enter to quit.")
    while True:
        user_input = input("Enter command: ")
        if user_input == 'r':
            if not is_recording:
                print("Starting recording...")
                start_recording()
            else:
                print("Stopping recording...")
                stop_recording()
                recording_finished.wait()  # Wait for recording to be saved
        elif user_input == 'q':
            if is_recording:
                stop_recording()
                recording_finished.wait()
                transcribe_audio(recording_file_name)
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
