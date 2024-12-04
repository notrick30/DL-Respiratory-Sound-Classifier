import numpy as np
from os import mkdir, remove
from os.path import exists, isdir

import audio_processing as ap
import heart_filtering as hf
import lung_filtering as lf

# Generate mel spectrograms for visualization
def generate_spectrograms(audio_data_list, sample_rate, titles):
    """
    Generates and displays mel spectrograms for a list of audio data arrays.

    Parameters:
        audio_data_list (list of numpy.ndarray): List of audio data arrays.
        sample_rate (int): Sample rate of the audio data.
        titles (list of str): Titles for each spectrogram.
    """
    for audio_data, title in zip(audio_data_list, titles):
        ap.generate_mel_spectrogram(audio_data, sample_rate, title=title)

def main():
    # Set this dictionary up in "folder_name" : ["file1", "file2", ...] format
    files = {
        'lung_samples': [ # "lung_samples" is the folder name
            'lung_rec_1', # These are the file names, without the `.wav` extension
            'lung_rec_2',
            'lung_rec_3',
            'lung_rec_4',
            'lung_rec_5',
            'lung_rec_6',
            'lung_rec_7'
        ]
    }

    output_folder = 'output' # Creates a folder of this name inside each input folder in `files` dictionary
    fileext = '.wav'

    for folder in files:
        outfolder = f'{folder}/{output_folder}'
        if exists(outfolder) and not isdir(outfolder):
            # Delete this non-directory file and create a directory
            remove(outfolder)
            mkdir(outfolder)
        elif not exists(outfolder):
            mkdir(outfolder)

        # Process all files in this folder
        for filename in files[folder]:
            input_filename = f'{folder}/{filename}{fileext}'
            output_filename = f'{outfolder}/{filename}_filtered{fileext}'

            # Read the input audio file
            audio_data, sample_rate = ap.read_wav_audio(input_filename)

            # Process the lung audio data
            cleaned_lung = lf.lung_filter(audio_data, sample_rate)
            cleaned_lung_normalized = ap.normalize_int16(cleaned_lung)

            # Process the heart audio data
            # cleaned_heart_normalized = hf.extract_heart_and_normalize(audio_data, sample_rate)

            # Save the cleaned lung sound to a WAV file
            ap.save_wav_audio(cleaned_lung_normalized, sample_rate, output_filename)

            print(f'Processed and saved: {output_filename}')

            # Prepare the data and titles
            audio_data_list = [audio_data, cleaned_lung]
            titles = ['Original Audio', 'Cleaned Lung Sound']

            generate_spectrograms(audio_data_list, sample_rate, titles)

if __name__ == '__main__':
    main()
