from flask import Flask, request, render_template
import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
from tempfile import NamedTemporaryFile

app = Flask(__name__)

# Load the saved model and label encoder
model_path = 'emotion_detection_model.h5'
label_encoder_path = 'label_encoder.pkl'
model = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

# Function to preprocess audio for prediction
def preprocess_audio(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    input_data = mfcc_scaled.reshape(1, mfcc_scaled.shape[0], 1)
    return input_data

# Function to make predictions
def predict_emotion(audio, sr):
    input_data = preprocess_audio(audio, sr)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.classes_[predicted_class]
    prediction_loss = 1 - np.max(prediction)  # Calculate input loss
    return predicted_label, prediction_loss

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['audio_file']
    if file.filename == '':
        return 'No selected file', 400

    with NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        audio, sr = librosa.load(temp.name, sr=None)

    predicted_emotion, prediction_loss = predict_emotion(audio, sr)
    return render_template('result.html', emotion=predicted_emotion, loss=prediction_loss)

@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        duration = int(request.form.get('duration', 5))
        sr = 16000
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        predicted_emotion, prediction_loss = predict_emotion(audio, sr)
        return render_template('result.html', emotion=predicted_emotion, loss=prediction_loss)

    return render_template('record.html')

if __name__ == "__main__":
    app.run(debug=True)
