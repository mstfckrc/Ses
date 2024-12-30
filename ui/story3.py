from tkinter import Tk, Toplevel, Label, Button, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict, Counter
import librosa
import numpy as np
from controller.emotion.predict import predict_emotion
from model.audio.train_speaker import train_speaker_model
import sounddevice as sd
import os
from config import SAMPLE_RATE, EMOTION_DATA_DIR
from model.emotion.emotion_train_speaker import train_emotion_model
from globals import get_speaker_model as getSpeakerModel, get_label_encoder as getLabelEncoder, get_scaler as getScaler , get_audio_accuracy as getAudioAccuracy, get_audio_f1_score as getAudioF1Score , get_emotion_accuracy as getEmotionAccuracy, get_emotion_f1_score as getEmotionF1Score

# Yeni eklenenler
audio_recording = []  # Kaydedilen sesi tutar
speaker_intervals = defaultdict(list)  # Konuşmacı zaman aralıkları
speaker_colors = {}  # Konuşmacı renkleri
SPEECH_THRESHOLD = 0.1  # Ses seviyesi eşiği

def user_story_3():
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
                        speaker_colors[speaker] = np.random.rand(3,)

                    # Zaman aralığı ekleme
                    speaker_intervals[speaker].append({
                        "start": i / SAMPLE_RATE,
                        "end": (i + step_size) / SAMPLE_RATE,
                        "emotion": emotion
                    })

                except Exception as e:
                    print(f"Hata: {e}")

        show_analysis_results()

    def show_analysis_results():
        """Analiz sonuçlarını ve grafikleri yeni bir pencerede gösterir."""
        analysis_window = Toplevel(root)
        analysis_window.title("Analiz Sonuçları")
        analysis_window.geometry("1200x900")
        analysis_window.configure(bg="white")  # Analiz penceresinin arka planı

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Sesin genliğini (seviyesini) hesapla ve zaman eksenine yerleştir
        time_axis = np.linspace(0, len(audio_recording) / SAMPLE_RATE, len(audio_recording))
        audio_amplitude = np.abs(audio_recording).flatten()

        ax1.plot(time_axis, audio_amplitude, color='purple', label='Ses Eşiği')  # Ses eşiği grafiği

        total_durations = defaultdict(float)  # Her konuşmacının toplam konuşma süresi
        emotion_counts = defaultdict(Counter)  # Her konuşmacının duygu dağılımı

        for speaker, intervals in speaker_intervals.items():
            for interval in intervals:
                start, end = interval["start"], interval["end"]
                emotion = interval["emotion"]

                # Zaman aralığını renklendir
                ax1.axvspan(start, end, color=speaker_colors[speaker], alpha=0.5)
                total_durations[speaker] += end - start

                # Duygu dağılımını güncelle
                emotion_counts[speaker][emotion] += end - start

        # Grafik ayarları
        ax1.set_xlabel("Zaman (saniye)")
        ax1.set_ylabel("Genlik (Ses Seviyesi)")
        ax1.set_title("Ses Eşiği Grafiği ve Konuşmacı/Duygu Zaman Aralıkları")

        # Pasta grafiği oluşturma (konuşmacı bazında süre)
        speakers = list(total_durations.keys())
        durations = list(total_durations.values())
        colors = [speaker_colors[speaker] for speaker in speakers]
        ax2.pie(durations, labels=speakers, autopct="%1.1f%%", colors=colors, startangle=90)
        ax2.set_title("Konuşmacı Süreleri Dağılımı")

        plt.tight_layout()

        # Canvas ekleme
        canvas = FigureCanvasTkAgg(fig, master=analysis_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Konuşmacı ve duygu yüzdelerini hesaplama
        emotion_percentages = {}
        for speaker, emotions in emotion_counts.items():
            total_time = sum(emotions.values())
            emotion_percentages[speaker] = {emotion: (duration / total_time) * 100 for emotion, duration in emotions.items()}

        # Duygu yüzdelerini metin olarak gösterme
        legend_text = "\n".join([
            f"{speaker}: " + ", ".join([f"{emotion} %{percentage:.1f}" for emotion, percentage in percentages.items()])
            for speaker, percentages in emotion_percentages.items()
        ])
        legend_label = Label(analysis_window, text=f"Duygu Dağılımı:\n{legend_text}", font=("Arial", 10), bg="white", justify="left")
        legend_label.pack(pady=10)

        # Skor etiketlerini ekleme
        accuracy_label = Label(analysis_window, text=f"Ses Model Doğruluğu: {getAudioAccuracy():.2f}", font=("Arial", 12), fg="green", bg="white")
        accuracy_label.pack(pady=5)

        f1_label = Label(analysis_window, text=f"Ses F1-Score: {getAudioF1Score():.2f}", font=("Arial", 12), fg="blue", bg="white")
        f1_label.pack(pady=5)

        accuracy_label = Label(analysis_window, text=f"Duygu Model Doğruluğu: {getEmotionAccuracy():.2f}", font=("Arial", 12), fg="green", bg="white")
        accuracy_label.pack(pady=5)

        f1_label = Label(analysis_window, text=f"Duygu F1-Score: {getEmotionF1Score():.2f}", font=("Arial", 12), fg="blue", bg="white")
        f1_label.pack(pady=5)

        # Grafiği kapatmak için plt.close()
        plt.close(fig)  # Bu satır, matplotlib grafiğini kapatır ve belleği temizler.

    # Arayüz butonları
    start_button = Button(root, text="Kaydı Başlat", command=start_recording, font=("Arial", 12), bg="white")
    start_button.pack(pady=20)

    root.mainloop()
