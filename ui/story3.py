from tkinter import Tk,  Label, Button, messagebox
from collections import defaultdict
import librosa
import numpy as np
from controller.audio.result import show_analysis_results
from controller.emotion.predict import predict_emotion
from model.audio.train_speaker import train_speaker_model
import sounddevice as sd
import os
from config import SAMPLE_RATE, EMOTION_DATA_DIR
from model.emotion.emotion_train_speaker import train_emotion_model
from globals import get_speaker_model as getSpeakerModel, get_label_encoder as getLabelEncoder, get_scaler as getScaler

# Yeni eklenenler
audio_recording = []  # Kaydedilen sesi tutar
speaker_intervals = defaultdict(list)  # Konuşmacı zaman aralıkları
speaker_colors = {}  # Konuşmacı renkleri
SPEECH_THRESHOLD = 0.1  # Ses seviyesi eşiği

def user_story_3(): # NOSONAR
    """Ses kaydı yapar, konuşmacıları ve duyguları analiz eder, zaman grafiğini gösterir."""

    # Model eğitimi
    if not train_speaker_model():
        messagebox.showerror("Hata", "Konuşmacı modeli eğitilemedi!")
        return

    speaker_model = getSpeakerModel()
    label_encoder = getLabelEncoder()
    scaler = getScaler()

    # Duygu modeli eğitimi
    emotion_model, emotion_label_encoder, emotion_scaler = train_emotion_model()

    # Tkinter ana penceresi
    root = Tk()
    root.title("Konuşmacı ve Duygu Tanıma")
    root.geometry("500x300")
    root.configure(bg="white")  # Ana pencerenin arka planı

    label_info = Label(root, text="Ses Kaydı: Başlat -> Analiz -> Grafikte Gösterim", font=("Arial", 12), bg="white")
    label_info.pack(pady=10)

    # Kayıt durumu etiketi
    recording_label = Label(root, text="", font=("Arial", 12), fg="red", bg="white")
    recording_label.pack(pady=10)

    def start_recording():
        """Belirli süre ses kaydı yapar."""
        global audio_recording

        # Kayıt durumunu gösteren mesaj
        recording_label.config(text="Kayıt yapılıyor, lütfen bekleyin...")
        root.update_idletasks()  # Mesajın hemen güncellenmesi için

        duration = 10  # Kaydedilecek süre (saniye)
        print("Ses kaydı başlatılıyor...")
        audio_recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print("Ses kaydı tamamlandı.")

        # Kayıt tamamlandıktan sonra mesajı kaldır
        recording_label.config(text="")
        analyze_audio()

    def analyze_audio():
        """Ses kaydını analiz eder, konuşmacı zaman aralıklarını ve duyguları belirler."""
        global speaker_intervals, speaker_colors
        speaker_intervals.clear()  # Önceki analizleri temizle
        time_step = 1  # Zaman aralığı (1 saniye)
        num_samples = len(audio_recording)
        step_size = SAMPLE_RATE * time_step

        for i in range(0, num_samples, int(step_size)):
            segment = audio_recording[i:i + int(step_size)].flatten()
            audio_amplitude = np.max(np.abs(segment))  # Sesin maksimum genliği
            if audio_amplitude > SPEECH_THRESHOLD:  # Ses seviyesi eşiği
                try:
                    # MFCC çıkarımı ve konuşmacı tahmini
                    mfccs = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_fft=1024, n_mfcc=20)
                    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
                    scaled_mfccs = scaler.transform(mfccs_mean)
                    prediction = speaker_model.predict(scaled_mfccs)
                    speaker = label_encoder.inverse_transform(prediction)[0]

                    # Duygu tahmini sadece duygu dosyası olan konuşmacılar için yapılır
                    speaker_emotion_dir = os.path.join(EMOTION_DATA_DIR, speaker)
                    if os.path.exists(speaker_emotion_dir) and os.listdir(speaker_emotion_dir):
                        emotion = predict_emotion(segment, emotion_model, emotion_label_encoder, emotion_scaler)
                    else:
                        emotion = "Bilinmiyor"

                    # Renk atama
                    if speaker not in speaker_colors:
                        # noinspection PyArgumentEqualDefault
                        speaker_colors[speaker] = np.random.default_rng().random(3) #NOSONAR

                    # Zaman aralığı ekleme
                    speaker_intervals[speaker].append({
                        "start": i / SAMPLE_RATE,
                        "end": (i + step_size) / SAMPLE_RATE,
                        "emotion": emotion
                    })

                except Exception as e:
                    print(f"Hata: {e}")

        show_analysis_results(audio_recording , speaker_intervals, speaker_colors)


    # Arayüz butonları
    start_button = Button(root, text="Kaydı Başlat", command=start_recording, font=("Arial", 12), bg="white")
    start_button.pack(pady=20)

    root.mainloop()