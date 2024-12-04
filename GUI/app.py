from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

from predictor import LungSoundPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MEL_FOLDER'] = 'static/mel_spectrograms'

# Initialize predictor
predictor = LungSoundPredictor(
    model_path='model/lung_sound_classifier.h5',
    scaler_path='model/scaler.pkl'
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MEL_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)
        
        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate Mel Spectrogram and predict conditions
            mel_image_path = os.path.join(app.config['MEL_FOLDER'], f"{filename}.png")
            predictions = predictor.predict(file_path, mel_image_path)

            # Display results
            return render_template(
                'index.html', predictions=predictions, mel_path=mel_image_path
            )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
