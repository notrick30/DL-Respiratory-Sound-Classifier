import numpy as np
from scipy.signal import butter, filtfilt
from yodel import filter as ft

import audio_processing as ap

def extract_heart_audio(audio_data, sample_rate):
    """
    Extracts heart sounds from the given audio data using Gaussian filtering,
    parametric equalization, and bandpass filtering.

    Parameters:
        audio_data (numpy.ndarray): The input audio data array.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        numpy.ndarray: The extracted heart sounds without normalization.
    """
    # Apply Gaussian filter to smooth the audio signal
    sigma = 25
    gauss_audio = ap.gaussian_filter(audio_data, sigma, truncate=6)
    
    # Parametric Equalization to enhance heart sound frequencies
    num_bands = 2
    band = [75, 241]
    Qfactor = [0.40, 0.40]
    db_gain = [16.0, -24.0]

    parameq = ft.ParametricEQ(sample_rate, num_bands)
    parameq.set_band(0, band[0], Qfactor[0], db_gain[0])
    parameq.set_band(1, band[1], Qfactor[1], db_gain[1])
    parameq_audio = np.empty_like(gauss_audio)
    parameq.process(gauss_audio, parameq_audio)

    # Highpass filter to remove frequencies below heart sounds
    lowcutoff = 50
    l_order = 6
    bl, al = butter(l_order, lowcutoff, btype='highpass', fs=sample_rate)
    parameq_hp_audio = filtfilt(bl, al, parameq_audio)

    # Lowpass filter to remove frequencies above heart sounds
    highcutoff = 270
    h_order = 12
    bh, ah = butter(h_order, highcutoff, btype='lowpass', fs=sample_rate)
    parameq_bp_audio = filtfilt(bh, ah, parameq_hp_audio)

    # Return the extracted heart sounds without normalization
    heart_audio = parameq_bp_audio

    return heart_audio

def extract_heart_and_normalize(audio_data, sample_rate):
    """
    Extracts heart sounds from the input WAV file, normalizes them,
    and saves them to the output WAV file.

    Parameters:
        input_filename (str): The path to the input WAV file.
        output_filename (str): The path to the output WAV file.
    """
    # Load the audio recording
    # audio_data, sample_rate = ap.read_wav_audio(input_filename)
    
    # Extract heart sounds
    heart_audio = extract_heart_audio(audio_data, sample_rate)
    
    # Normalize the heart audio
    normalized_heart_audio = ap.normalize_int16(heart_audio)
    
    # Save the normalized heart audio to a WAV file
    # ap.save_wav_audio(normalized_heart_audio, sample_rate, output_filename)
    
    return normalized_heart_audio
