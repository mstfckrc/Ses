import os
from tkinter import messagebox, simpledialog
import sounddevice as sd
from config import DURATION, EMOTION_DATA_DIR, SAMPLE_RATE
import scipy.io.wavfile as wavfile

def record_user_emotion():
    """Kullanıcıdan duygu kaydı alır ve kaydeder."""
    
    # Duygular için kayıt dizini
    emotion_dir = os.path.join(EMOTION_DATA_DIR)
    os.makedirs(emotion_dir, exist_ok=True)
    
    speaker_name = simpledialog.askstring("Konuşmacı Adı", "Konuşmacının adını girin:")
    if not speaker_name:
        messagebox.showerror("Hata", "Konuşmacı adı gerekli!")
        return

    speaker_dir = os.path.join(emotion_dir, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)

    emotions = ["Mutlu", "Üzgün", "Sinirli"]

    for i, emotion in enumerate(emotions):
        messagebox.showinfo(
            "Duygu Kaydı Bilgisi", 
            f"{emotion} duygusu için kayda hazır olun. 'Tamam' butonuna tıklayın ve konuşmaya başlayın."
        )
        # Ses kaydını başlat
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Kayıt bitene kadar bekle

        # Dosya yolunu oluştur ve kaydet
        file_path = os.path.join(speaker_dir, f"{speaker_name}_{emotion}.wav")
        wavfile.write(file_path, SAMPLE_RATE, audio)
        print(f"Duygu kaydı kaydedildi: {file_path}")

    messagebox.showinfo("Tamamlandı", f"{speaker_name} için duygu kayıtları tamamlandı!")
