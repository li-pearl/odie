import pyaudio
import wave
import speech_recognition as sr
import keyboard
import time
import io

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Buffer size
OUTPUT_FILENAME = "recorded_audio.wav"

def record_speech(button="space"):
    audio_interface = pyaudio.PyAudio()

    print(f"Press and hold '{button}' to record speech...")

    # Wait until the button is pressed
    while not keyboard.is_pressed(button):
        time.sleep(0.01)  # Prevent CPU overuse

    print("Recording... Speak now.")

    stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                                  rate=RATE, input=True,
                                  frames_per_buffer=CHUNK)

    frames = []

    # Continuously capture audio while button is held
    while keyboard.is_pressed(button):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Stopped recording. Processing...")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    # Save recorded audio to a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {OUTPUT_FILENAME}")

    # Convert speech to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)  # Read entire audio file
        try:
            text = recognizer.recognize_google(audio_data)
            print("Recognized Text:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Speech recognition service unavailable.")
            return ""

if __name__ == "__main__":
    result = record_speech()
    print("Final Output:", result)