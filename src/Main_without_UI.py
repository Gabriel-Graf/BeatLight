import time

import pandas as pd
import wave
import numpy as np

import customDataloader
from ResolumeBridge import activate_layer

from _spinner_helper import Spinner
import pyaudiowpatch as pyaudio
import onnxruntime as ort

# Aufnahmeparameter
FORMAT = pyaudio.paInt16
CHUNK = 1024
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = "recorded_audio/recorded_audio.wav"
DEVICE_INDEX = 16
MODEL_DIR_PATH = "saved_models"
dataset_path = r"Dataset/metadata.csv"
data_path = r"Dataset/samples"
df = pd.read_csv(dataset_path, encoding='latin')
dl = customDataloader.SoundDSMel(df, data_path)


def get_default_speakers():
    """Get the default WASAPI loopback device for audio recording."""
    with pyaudio.PyAudio() as p, Spinner() as spinner:
        print("Searching for default loopback output device...")
        try:
            # Get default WASAPI info
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            spinner.print("Looks like WASAPI is not available on the system. Exiting...")
            spinner.stop()
            exit()

        # Get default WASAPI speakers
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                spinner.print(
                    "Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                spinner.stop()
                exit()
        else:
            spinner.print("Default loopback output device found.")

    return default_speakers


def record_audio(default_speakers):
    """Record audio from the default WASAPI loopback device and save it to a WAV file."""
    with pyaudio.PyAudio() as p, Spinner() as spinner:
        spinner.print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}")

        wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        with p.open(format=pyaudio.paInt16,
                    channels=default_speakers["maxInputChannels"],
                    rate=int(default_speakers["defaultSampleRate"]),
                    frames_per_buffer=CHUNK,
                    input=True,
                    input_device_index=default_speakers["index"],
                    stream_callback=callback
                    ) as stream:
            spinner.print(f"The next {RECORD_SECONDS} seconds will be written to {WAVE_OUTPUT_FILENAME}")
            time.sleep(RECORD_SECONDS)  # Blocking execution while playing

        wave_file.close()


def convert_to_melspec(filename):
    """Convert a recorded audio file to a Mel-spectrogram tensor for further processing of the model."""
    audio, sr = dl.load_wave(filename)
    audio, sr = dl.wave_to_mono(audio, sr)

    # Mel-Spektrogramm erzeugen
    mel_spec = dl.wave_to_melspec(audio, sr, n_fft=4096, hop_length=128, n_mels=256)
    pad_mel = mel_spec[..., :10336]  # Optionales Padding
    normalized_mel_spec = dl.normalize_and_center_melspec(pad_mel)
    mel_tensor = normalized_mel_spec.float()

    return mel_tensor


def predict(mel_spec, model_path=f"{MODEL_DIR_PATH}/rock_metal.onnx"):
    """Predict the class of a Mel-spectrogram using an ONNX model."""
    ort_session = ort.InferenceSession(model_path)
    mel_tensor = np.expand_dims(np.array(mel_spec, dtype=np.float32), axis=0)

    outputs = ort_session.run(None, {"input": mel_tensor})
    output = outputs[0]
    print("Prediction: ", output, " - Label: ", list(dl.label_dict.keys())[np.argmax(output)])

    return list(dl.label_dict.keys())[np.argmax(output)]


def set_resolume_visual(label):
    """Set the visual in Resolume based on the predicted label."""
    if label == "Rock'n'Roll":
        activate_layer(1, 1)
        activate_layer(2, 1)
        activate_layer(3, 1)
    elif label == "HardRock":
        activate_layer(1, 2)
        activate_layer(2, 2)
        activate_layer(3, 2)
    elif label == "HeavyMetal":
        activate_layer(1, 3)
        activate_layer(2, 3)
        activate_layer(3, 3)
    elif label == "TrashMetal":
        activate_layer(1, 4)
        activate_layer(2, 4)
        activate_layer(3, 4)
    elif label == "MetalcoreMelodic":
        activate_layer(1, 5)
        activate_layer(2, 5)
        activate_layer(3, 5)
    elif label in "DeathMetal":
        activate_layer(1, 6)
        activate_layer(2, 6)
        activate_layer(3, 6)
    elif label == "Grindcore":
        activate_layer(1, 7)
        activate_layer(2, 7)
        activate_layer(3, 7)


if __name__ == "__main__":
    default_speakers = get_default_speakers()
    record_audio(default_speakers)
    mel_spec = convert_to_melspec(WAVE_OUTPUT_FILENAME)
    lable = predict(mel_spec)
    set_resolume_visual(lable)
