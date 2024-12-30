import os
from tkinter import messagebox, simpledialog
import sounddevice as sd
from config import DATA_DIR, DURATION, SAMPLE_RATE
import scipy.io.wavfile as wavfile

def record_user_voice():
    """Kullanıcıdan ses kaydı alır ve kaydeder."""
    global ground_truth

    speaker_name = simpledialog.askstring("Konuşmacı Adı", "Konuşmacının adını girin:")
    if not speaker_name:
        messagebox.showerror("Hata", "Konuşmacı adı gerekli!")
        return

    speaker_dir = os.path.join(DATA_DIR, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)

    ground_truth = speaker_name  # Gerçek konuşmacı kimliği kaydedilir

    for i in range(3):  # 3 farklı kayıt al
        messagebox.showinfo("Kayıt Bilgisi", f"{i + 1}. kayda hazır olun. 'Tamam' butonuna tıklayın ve konuşmaya başlayın.")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Kayıt bitene kadar bekle
        file_path = os.path.join(speaker_dir, f"{speaker_name}_{i + 1}.wav")
        wavfile.write(file_path, SAMPLE_RATE, audio)
        print(f"Kayıt kaydedildi: {file_path}")

    messagebox.showinfo("Tamamlandı", f"{speaker_name} için ses kayıtları tamamlandı!")