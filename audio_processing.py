import numpy as np
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Function to read raw audio data and convert to numpy array
def read_raw_audio(file_path):
    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
    return np.frombuffer(raw_data, dtype=np.int16)

# Function to save numpy audio data to a WAV file
def save_wav_audio(audio_data, sample_rate, output_path):
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.astype(np.int16).tobytes())

# Function to read mono WAV file saved by convert_to_wav()
def read_wav_audio(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Extract audio parameters
        num_channels = wav_file.getnchannels()  # expect to be 1 channel (mono)
        sample_width = wav_file.getsampwidth()  # expect to be 2 bytes
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        # Read and convert the frames to a NumPy array
        frames = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data, sample_rate

def normalize_int16(audio_data):
    # Calculate the peak amplitude of the audio data
    peak = np.max(np.abs(audio_data))
    if peak == 0:
        # Avoid division by zero if the audio is silent
        return audio_data
    
    # Scale the audio data to the full range of int16
    normalized_audio = audio_data * (np.iinfo(np.int16).max / peak)
    return normalized_audio.astype(np.int16)  # Convert back to 16-bit PCM

def amplify_by_db(audio_data, db_increase):
    # Convert dB increase to a linear factor
    amplification_factor = 10 ** (db_increase / 20)
    # Amplify the audio data
    amplified_audio = audio_data * amplification_factor
    # Ensure it doesn't exceed the int16 range
    amplified_audio = np.clip(amplified_audio, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return amplified_audio.astype(np.int16)

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Applying gaussian filter over audio_data
def gaussian_filter(audio_data, sigma=1.0, truncate=4):
    # test the sigma value out and settle on one
    gaussfiltered_audio = gaussian_filter1d(audio_data, sigma, truncate=truncate)
    return gaussfiltered_audio

# Function to generate a Mel spectrogram
def generate_mel_spectrogram(audio_data, sample_rate, title='Mel Spectrogram'):
    # Generate and display the Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_data.astype(float), sr=sample_rate, n_mels=128, fmax=2000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', fmax=2000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return S
