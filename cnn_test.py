import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

class LungSoundPredictor:
    def __init__(self, model_path='lung_sound_classifier.h5', scaler_path='scaler.pkl'):
        self.label_mapping = {'wheeze': 0, 'stridor': 1, 'rhonchi': 2, 'crackles': 3}
        
        # Load the trained model
        self.model = load_model(model_path)
        
        # Load the fitted scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
    def audio_to_mel_spectrogram(self, file_path, n_mels=128, duration=15):
        try:
            # Load audio with fixed duration
            y, sr = librosa.load(file_path, sr=22050, duration=duration)
            
            # Pad if audio is too short
            if len(y) < duration * sr:
                y = np.pad(y, (0, duration * sr - len(y)))
                
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Plot the mel spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.show()
            
            return mel_spec_db
        
        except Exception as e:
            logging.error(f'Error processing file {file_path}: {str(e)}')
            return None

    def predict(self, file_path, threshold=0.5):
        try:
            # Process audio file
            mel_spec = self.audio_to_mel_spectrogram(file_path)
            if mel_spec is None:
                raise ValueError('Could not process audio file')
                
            # Normalize spectrogram
            mel_spec_flat = mel_spec.reshape(1, -1)
            mel_spec_scaled = self.scaler.transform(mel_spec_flat)
            mel_spec = mel_spec_scaled.reshape(mel_spec.shape)
            
            # Add dimensions for model
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Get predictions
            predictions = self.model.predict(mel_spec)[0]
            
            # Convert to labels
            detected_labels = []
            for i, prob in enumerate(predictions):
                if prob >= threshold:
                    detected_labels.append(list(self.label_mapping.keys())[i])
                    
            return detected_labels if detected_labels else ['Normal']
            
        except Exception as e:
            logging.error(f'Error during prediction: {str(e)}')
            raise

# Example usage for prediction
if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = LungSoundPredictor()
        
        # Make prediction for a specific file
        test_file = 'C:/Users/sudar/OneDrive/Desktop/DL-Respiratory-Sound-Classifier/HF_Lung_V1-master/train/steth_20190626_15_12_23.wav'
        result = predictor.predict(test_file)
        print(f'Detected conditions: {result}')
        
    except Exception as e:
        logging.error(f'Program error: {str(e)}')
