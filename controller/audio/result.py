from collections import defaultdict
from tkinter import Toplevel, Label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict, Counter
import numpy as np
from config import SAMPLE_RATE
from globals import get_root , get_audio_accuracy as getAudioAccuracy , get_audio_f1_score as getAudioF1Score , get_emotion_accuracy as getEmotionAccuracy , get_emotion_f1_score as getEmotionF1Score


def show_analysis_results(audio_recording, speaker_intervals, speaker_colors):
        """Analiz sonuçlarını ve grafikleri yeni bir pencerede gösterir."""
        root = get_root()
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