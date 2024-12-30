from tkinter import messagebox
import librosa
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from config import DATA_DIR, SAMPLE_RATE
from globals import get_speaker_model as getSpeakerModel, get_label_encoder as getLabelEncoder, get_scaler as getScaler , set_audio_accuracy as setAudioAccuracy, set_audio_f1_score as setAudioF1Score , set_label_encoder as setLabelEncoder , set_speaker_model as setSpeakerModel
from model.audio.speakers_data import load_speakers_data

def train_speaker_model():
    """Konuşmacı tanıma modeli eğitir."""
    scaler = getScaler()

    speakers_data = load_speakers_data(DATA_DIR)
    if not speakers_data:
        messagebox.showerror("Hata", "Ses verisi bulunamadı!")
        return False

    # Özellik çıkarımı ve etiketleme
    X = []
    y = []
    for speaker, files in speakers_data.items():
        for file in files:
            audio, sr = librosa.load(file, sr=SAMPLE_RATE)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            X.append(mfccs_mean)
            y.append(speaker)

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    setLabelEncoder(label_encoder)
    y_encoded = label_encoder.fit_transform(y)

    # Veriyi normalize et (Scaler eğitimde fit edilir)
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Model eğitimi
    speaker_model = SVC(kernel="linear", probability=True)
    speaker_model.fit(x_train, y_train)
    setSpeakerModel(speaker_model)

    # Model doğruluğunu hesapla
    y_pred = speaker_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    setAudioAccuracy(accuracy)
    setAudioF1Score(f1)
    print(f"Model doğruluğu: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    return True