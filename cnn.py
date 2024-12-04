import librosa
import logging
import numpy as np
import pickle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
import os
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LungSoundClassifier:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.label_mapping = {'wheeze': 0, 'stridor': 1, 'rhonchi': 2, 'crackles': 3}
        self.model = None
        self.scaler = StandardScaler()
        
    def audio_to_mel_spectrogram(self, file_path, n_mels=128, duration=15):
        try:
            # Load audio with fixed duration
            y, sr = librosa.load(file_path, sr=22050, duration=duration)
            
            # Pad if audio is too short
            if len(y) < duration * sr:
                y = np.pad(y, (0, duration * sr - len(y)))
                
            # Generate mel spectrogram with fixed parameters
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=512
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_spec_db
        
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {str(e)}')
            return None

    def parse_label_file(self, label_file):
        try:
            if not os.path.exists(label_file):
                logger.warning(f'Label file not found: {label_file}')
                return None

            with open(label_file, 'r') as f:
                lines = f.readlines()

            label_vector = [0] * len(self.label_mapping)
            
            for line in lines:
                line = line.strip() # Remove extra spaces
                tokens = line.split() # Tokenize line into words
                for token in tokens:
                    if token == 'D': # Only detect standalone uppercase D
                        label_vector[self.label_mapping['crackles']] = 1
                    elif token.lower() == 'wheeze':
                        label_vector[self.label_mapping['wheeze']] = 1
                    elif token.lower() == 'stridor':
                        label_vector[self.label_mapping['stridor']] = 1
                    elif token.lower() == 'rhonchi':
                        label_vector[self.label_mapping['rhonchi']] = 1

            return label_vector
        
        except Exception as e:
            logger.error(f'Error parsing label file {label_file}: {str(e)}')
            return None

    def load_data(self, folder_path):
        X = []
        y = []
        files_processed = 0
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                label_file = file_path.replace('.wav', '_label.txt')
                
                # Process audio file
                mel_spec = self.audio_to_mel_spectrogram(file_path)
                if mel_spec is None:
                    continue
                    
                # Process label file
                label_vector = self.parse_label_file(label_file)
                if label_vector is None:
                    continue
                
                X.append(mel_spec)
                y.append(label_vector)
                files_processed += 1
                
                if files_processed % 10 == 0:
                    logger.info(f'Processed {files_processed} files from {folder_path}')

        if not X:
            raise ValueError(f'No valid files found in {folder_path}')

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.label_mapping), activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss=BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        return model

    def train(self, epochs=20, batch_size=32):
        try:
            # Load and preprocess data
            logger.info('Loading training data...')
            X_train, y_train = self.load_data(self.train_path)
            logger.info('Loading test data...')
            X_test, y_test = self.load_data(self.test_path)
            
            # Normalize spectrograms
            X_train_flat = X_train.reshape((X_train.shape[0], -1))
            X_test_flat = X_test.reshape((X_test.shape[0], -1))
            
            X_train_scaled = self.scaler.fit_transform(X_train_flat)
            X_test_scaled = self.scaler.transform(X_test_flat)
            
            X_train = X_train_scaled.reshape(X_train.shape)
            X_test = X_test_scaled.reshape(X_test.shape)
            
            # Add channel dimension
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            
            # Build and train model
            logger.info(f'Building model with input shape {X_train.shape[1:]}')
            self.model = self.build_model(X_train.shape[1:])
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Save model and scaler
            self.model.save('lung_sound_classifier.h5')
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info('Model and scaler saved successfully')
            
            return history
            
        except Exception as e:
            logger.error(f'Error during training: {str(e)}')
            raise

# Run training
if __name__ == '__main__':
    try:
        # Initialize classifier with your paths
        classifier = LungSoundClassifier(
            train_path='C:/Users/sudar/OneDrive/Desktop/DL-Respiratory-Sound-Classifier/HF_Lung_V1-master/train',
            test_path='C:/Users/sudar/OneDrive/Desktop/DL-Respiratory-Sound-Classifier/HF_Lung_V1-master/test'
        )
        
        # Train model
        history = classifier.train(epochs=20, batch_size=32)
        
    except Exception as e:
        logger.error(f'Program error: {str(e)}')
