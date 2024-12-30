import librosa
import numpy as np

def predict_emotion(audio_segment, model, label_encoder, scaler):
    """Bir ses segmentinin duygusunu tahmin eder."""
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=16000, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
    scaled_mfccs = scaler.transform(mfccs_mean)
    prediction = model.predict(scaled_mfccs)
    emotion = label_encoder.inverse_transform(prediction)[0]
    return emotion
