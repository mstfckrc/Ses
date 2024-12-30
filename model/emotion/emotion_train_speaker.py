from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import librosa
import numpy as np
from globals import set_emotion_accuracy as setEmotionAccuracy, set_emotion_f1_score as setEmotionF1Score
from model.emotion.emotion_speakers_data import load_emotions_data

def train_emotion_model():
    """Duygu tanıma modeli eğitir."""
    data_dir = "data/emotion_records"
    emotions_data = load_emotions_data(data_dir)
    if not emotions_data:
        print("Duygu verisi bulunamadı!")
        return False

    X = []
    y = []
    for speaker, files in emotions_data.items():
        for file, emotion in files:
            audio, sr = librosa.load(file, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            X.append(mfccs_mean)
            y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = SVC(kernel="linear", probability=True, C=10, gamma='scale' , random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    setEmotionAccuracy(accuracy)
    setEmotionF1Score(f1)
    
    print(f"Model doğruluğu: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")

    return model, label_encoder, scaler
