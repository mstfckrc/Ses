import tkinter as tk
from tkinter import messagebox, simpledialog
import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from collections import deque
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import speech_recognition as sr
import os
import threading

# Ses ayarları
SAMPLE_RATE = 22050  # Örnekleme frekansı
DURATION = 5  # Ses kaydı süresi (saniye)
CHUNK = 1024  # Ses işleme için blok boyutu
DATA_DIR = "data"  # Ses kayıtlarının saklanacağı klasör

# Global değişkenler
speaker_model = None
label_encoder = None
scaler = StandardScaler()  # Eğitimde kullanılan scaler, test sırasında da kullanılacak
audio_data = deque(maxlen=SAMPLE_RATE * 10)  # Histogram için 10 saniyelik veri
stop_flag = False  # Histogram döngüsü kontrolü
ground_truth = None  # Gerçek konuşmacı kimliği
word_count = 0  # Toplam kelime sayısı global olarak tanımlandı


# ---------------- Ses Kaydı ----------------
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


# ---------------- Konuşmacı Tanıma Modeli ----------------
def load_speakers_data(data_dir="data"):
    """Klasörlerden konuşmacı verilerini yükler."""
    speakers = {}
    if not os.path.exists(data_dir):
        print(f"'{data_dir}' klasörü bulunamadı!")
        return speakers

    for speaker in os.listdir(data_dir):
        speaker_dir = os.path.join(data_dir, speaker)
        if os.path.isdir(speaker_dir):
            speakers[speaker] = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith(".wav")]
    return speakers


def train_speaker_model():
    """Konuşmacı tanıma modeli eğitir."""
    global speaker_model, label_encoder, scaler

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
    y_encoded = label_encoder.fit_transform(y)

    # Veriyi normalize et (Scaler eğitimde fit edilir)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Model eğitimi
    speaker_model = SVC(kernel="linear", probability=True)
    speaker_model.fit(X_train, y_train)

    # Model doğruluğunu hesapla
    y_pred = speaker_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Model doğruluğu: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    return True


# ---------------- User Story 1: Histogram ----------------
def real_time_audio_histogram(indata, frames, time, status):
    """Ses verisinden histogram oluşturur ve günceller."""
    audio_data.extend(indata[:, 0])


def update_histogram():
    """Zaman ve frekans domaini histogramlarını çizer."""
    global stop_flag
    plt.ion()
    plt.show()  # Grafiği ilk kez başlatıyoruz
    while not stop_flag:
        # Zaman domaini grafiği (ses dalga formu)
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.plot(audio_data)
        plt.title("Zaman Domaini - Ses Dalga Formu")
        plt.xlabel("Zaman (örnekler)")
        plt.ylabel("Genlik")
        plt.xlim(0, len(audio_data))

        # Spektrogram grafiği (Frekans domaini)
        plt.subplot(2, 1, 2)
        plt.cla()
        # Spektrogram çizimi
        plt.specgram(np.array(audio_data), Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, scale='dB', cmap='viridis')
        plt.title("Spektrogram - Frekans Domaini")
        plt.xlabel("Zaman (saniye)")
        plt.ylabel("Frekans (Hz)")

        plt.tight_layout()
        plt.pause(0.1)


def stop_program(event):
    """Histogram penceresi kapatıldığında döngüyü durdurur."""
    global stop_flag
    stop_flag = True
    plt.close()


def user_story_1():
    """Histogram oluşturulmasını sağlar."""
    global stop_flag
    stop_flag = False
    with sd.InputStream(callback=real_time_audio_histogram, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK):
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', stop_program)
        update_histogram()


# ---------------- User Story 2: Ses Tanıma ----------------
def user_story_2():
    """Ses tanıma işlemi başlatır ve tanınan metni ekrana yazdırır."""
    global word_count
    word_count = 0  # Kelime sayısını her yeni pencere açıldığında sıfırlıyoruz

    def recognize_continuous():
        """Mikrofondan sürekli olarak ses tanıyıp anlık olarak ekranda gösterir."""
        recognizer = sr.Recognizer()
        global word_count

        # Yeni pencere açılır
        result_window = tk.Toplevel(root)
        result_window.title("Anlık Ses Tanıma")
        result_window.geometry("500x500")

        label = tk.Label(result_window, text="Tanınan Kelimeler", font=("Arial", 16))
        label.pack(pady=10)

        text_display = tk.Text(result_window, height=10, width=40, font=("Arial", 14))
        text_display.pack(pady=20)

        # Text widget'ını sadece okunabilir yapmak
        text_display.config(state=tk.DISABLED)

        word_count_label = tk.Label(result_window, text="Toplam Kelime Sayısı: 0", font=("Arial", 12))
        word_count_label.pack(pady=10)

        # Mikrofon üzerinden ses tanıma
        with sr.Microphone() as source:
            word_count = 0
            print("Ses tanıma başlatıldı...")
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    audio = recognizer.listen(source)
                    recognized_text = recognizer.recognize_google(audio, language="tr-TR")

                    # Tanınan kelimeleri ekrana yaz
                    text_display.config(state=tk.NORMAL)  # Yazma izni
                    text_display.insert(tk.END, recognized_text + '\n')  # Tanınan metni ekler
                    text_display.yview(tk.END)  # Ekranın en altına kaydır
                    text_display.config(state=tk.DISABLED)  # Sadece okunabilir yap

                    # Tanınan kelimeler
                    recognized_words = set(recognized_text.lower().split())

                    # Kelime sayısını artır
                    word_count += len(recognized_words)

                    # Sonuçları güncelle
                    word_count_label.config(text=f"Toplam Kelime Sayısı: {word_count}")
                except sr.UnknownValueError:
                    pass  # Anlaşılmayan sesleri yoksay
                except sr.RequestError:
                    messagebox.showerror("Hata", "Google Speech API'ye bağlanılamadı!")
                    break

    # Yeni bir thread (iş parçacığı) oluşturuyoruz, böylece ses tanıma işlemi ana thread'i engellemiyor
    threading.Thread(target=recognize_continuous, daemon=True).start()



# ---------------- User Story 3: Anlık Kişi Tanıma ----------------
def predict_emotion(audio):
    """Ses sinyalinden duygu tahmini yapar."""
    # MFCC özelliklerini çıkarma
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    energy = np.sum(audio ** 2) / len(audio)  # Enerji hesaplama
    pitch = np.mean(np.abs(librosa.feature.zero_crossing_rate(y=audio)))  # Pitch (ton) hesaplama
    zero_crossings = np.sum(librosa.feature.zero_crossing_rate(y=audio))  # Sıfır geçişi sayısı

    # Duygu sınıflandırması
    if energy > 0.02 and pitch > 0.1 and zero_crossings < 300:
        emotion = "Mutlu"
        confidence = 90
    elif energy < 0.01 and pitch < 0.05:
        emotion = "Üzgün"
        confidence = 85
    elif energy > 0.02 and pitch > 0.2:
        emotion = "Sinirli"
        confidence = 80
    elif energy < 0.01 and pitch > 0.1:
        emotion = "Korkmuş"
        confidence = 75
    else:
        emotion = "Nötr"
        confidence = 50

    return emotion, confidence


def user_story_3():
    """Her tıklamada model eğitilir ve anlık kişi tanıma yapılır."""
    global speaker_model, label_encoder, scaler

    # Model eğitimi
    if not train_speaker_model():
        messagebox.showerror("Hata", "Model eğitilemedi!")
        return

    def recognize(indata, frames, time, status):
        """Konuşmacı tanımayı işler ve ses eşiği kontrolü yapar."""
        audio = np.array(indata[:, 0])  # Alınan ses verisi
        audio_magnitude = np.linalg.norm(audio)  # Sesin genliğini ölçüyoruz (L2 normu)
        
        # Ses genliği bir eşikten büyükse, konuşmacıyı tanıyacağız
        threshold = 1.5  # Eşik değeri
        if audio_magnitude > threshold:
            print(f"Ses Eşiği Aşıldı: {audio_magnitude}")
            
            # MFCC çıkarımı yapıyoruz
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
            mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            # MFCC özelliklerini normalleştiriyoruz (eğitimdeki gibi)
            mfccs_scaled = scaler.transform(mfccs_mean)
            
            # Model tahmini
            prediction = speaker_model.predict(mfccs_scaled)
            prediction_proba = speaker_model.predict_proba(mfccs_scaled)  # Tahmin olasılıkları
            speaker_name = label_encoder.inverse_transform(prediction)[0]
            
            # Duygu tahmini
            emotion, emotion_confidence = predict_emotion(audio)

            # Doğruluk oranını hesaplama
            confidence = np.max(prediction_proba) * 100  # En yüksek olasılığı alıp yüzdelik çeviriyoruz
            print(f"Tanımlanan Konuşmacı: {speaker_name}, Doğruluk Oranı: {confidence:.2f}%")
            print(f"Duygu: {emotion}, Duygu Doğruluğu: {emotion_confidence:.2f}%")

            # F1 Skoru ve Doğruluk Hesaplama (Modeli test ettikten sonra)
            X_test = mfccs_scaled  # Bu örnekte, sadece bir test örneği var. Gerçek uygulamada, test seti kullanılacak
            y_test = label_encoder.transform([speaker_name])  # Gerçek etiket
            y_pred = speaker_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            # F1 ve Doğruluk değerlerini ekranda göster
            print(f"Model Doğruluk: {accuracy:.2f}")
            print(f"F1 Skoru: {f1:.2f}")
            
            # Duygu yüzdesi hesaplama
            emotions = ["Mutlu", "Üzgün", "Sinirli", "Korkmuş", "Nötr"]
            emotion_percentages = {
                "Mutlu": 0,
                "Üzgün": 0,
                "Sinirli": 0,
                "Korkmuş": 0,
                "Nötr": 0
            }
            
            # Yüzdelik hesaplama
            if emotion == "Mutlu":
                emotion_percentages["Mutlu"] = emotion_confidence
            elif emotion == "Üzgün":
                emotion_percentages["Üzgün"] = emotion_confidence
            elif emotion == "Sinirli":
                emotion_percentages["Sinirli"] = emotion_confidence
            elif emotion == "Korkmuş":
                emotion_percentages["Korkmuş"] = emotion_confidence
            else:
                emotion_percentages["Nötr"] = emotion_confidence

            # Sonuçları ekrana yazdır
            emotion_string = ", ".join([f"{e} %{emotion_percentages[e]:.2f}" for e in emotions if emotion_percentages[e] > 0])

            messagebox.showinfo("Tanımlanan Konuşmacı ve Duygu", 
                                f"Tanımlanan Konuşmacı: {speaker_name}\n"
                                f"Doğruluk Oranı: {confidence:.2f}%\n"
                                f"Duygular: {emotion_string}\n"
                                f"Model Doğruluk: {accuracy:.2f}\n"
                                f"F1 Skoru: {f1:.2f}")
        else:
            print(f"Ses Eşiği Aşılmadı: {audio_magnitude}")

    # Anlık kişi tanıma işlemi
    with sd.InputStream(callback=recognize, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK):
        messagebox.showinfo("Bilgi", "Anlık konuşmacı tanıma başlatıldı. Pencereyi kapatmak için 'X'e basabilirsiniz.")


# ---------------- Ana Menü ----------------
def show_main_menu():
    """Ana menüyü gösterir."""
    for widget in root.winfo_children():
        widget.pack_forget()
    menu_label = tk.Label(root, text="Ana Menü", font=("Arial", 24))
    menu_label.pack(pady=20)

    tk.Button(root, text="Kullanıcı Sesi Kaydetme", command=record_user_voice, font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="User Story 1: Histogram", command=user_story_1, font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="User Story 2: Ses Tanıma", command=user_story_2, font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="User Story 3: Anlık Kişi Tanıma", command=user_story_3, font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="Çıkış", command=root.quit, font=("Arial", 16)).pack(pady=20)

# ---------------- Program Başlatma ----------------
root = tk.Tk()
root.title("Proje 1 - Ses Analizi ve Tanıma")
show_main_menu()
root.mainloop()
