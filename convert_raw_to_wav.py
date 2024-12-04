import audio_processing as ap

import numpy as np
import wave

rawfname = "recorded_audio.raw"
wav_savefname = "recorded_audio.wav"
sr = 16000

def read_raw_audio(file_path):
    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
    return np.frombuffer(raw_data, dtype=np.int16)

def save_wav_audio(audio_data, sample_rate, output_path):
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.astype(np.int16).tobytes())

audio = read_raw_audio(rawfname)
save_wav_audio(audio, sr, wav_savefname)
