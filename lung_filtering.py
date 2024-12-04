import heart_filtering as hf
import numpy as np
from scipy.signal import butter, filtfilt

import audio_processing as ap

def lung_filter(audio_data, sample_rate):
    """
    Processes the lung audio data by applying Gaussian filtering,
    bandpass filtering, and heart sound subtraction.

    Parameters:
        audio_data (numpy.ndarray): The input audio data array.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        numpy.ndarray: The cleaned lung sound audio data.
    """
    # Apply a lighter Gaussian filter to the lung sounds
    sigma_lung = 5  # Lighter sigma for lung sounds
    gauss_lung_audio = ap.gaussian_filter(audio_data, sigma=sigma_lung, truncate=6)

    # Filter ambient noise to focus on lung sounds (100 Hz to 2000 Hz)
    lowcut = 100
    highcut = 2000
    order = 6
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=sample_rate)
    lung_bandpassed = filtfilt(b, a, gauss_lung_audio)

    # Extract heart sounds from the original audio data
    heart_sounds = hf.extract_heart_audio(audio_data, sample_rate)

    # Subtract the heart sounds from the lung bandpassed signal
    cleaned_lung = lung_bandpassed - heart_sounds

    return cleaned_lung
