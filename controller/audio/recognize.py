from tkinter import messagebox
import speech_recognition as sr
import tkinter as tk
from tkinter import ttk
from controller.topics.get_topics import get_topics_from_keywords
from globals import get_root as getRoot, word_count

def wrap_text(text, max_length):
    """Metni belirtilen uzunlukta alt satıra böler."""
    words = text.split()
    wrapped_lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:  # Maksimum uzunluğu aşıyorsa
            wrapped_lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:  # Kalan kelimeler
        wrapped_lines.append(" ".join(current_line))

    return wrapped_lines

def recognize_continuous():
    """Mikrofondan sürekli olarak ses tanıyıp anlık olarak ekranda gösterir."""
    header = "Cümle"
    recognizer = sr.Recognizer()
    global word_count
    word_count = 0

    root = getRoot()

    # Yeni pencere açılır
    result_window = tk.Toplevel(root)
    result_window.title("Anlık Ses Tanıma")
    result_window.geometry("600x500")

    # Tanınan kelimeler başlığı
    label = tk.Label(result_window, text="Tanınan Kelimeler", font=("Arial", 16))
    label.pack(pady=10)

    # Kelime sayısını gösteren etiket
    word_count_label = tk.Label(result_window, text="Toplam Kelime Sayısı: 0", font=("Arial", 12))
    word_count_label.pack(pady=10)

    # Tablo çerçevesi ve kaydırma çubuğu
    table_frame = tk.Frame(result_window)
    table_frame.pack(fill=tk.BOTH, expand=True)

    tree_scroll = ttk.Scrollbar(table_frame, orient="vertical")
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(
        table_frame,
        columns=(header, "Konu"),
        show="headings",
        yscrollcommand=tree_scroll.set,
        height=20
    )
    tree.pack(fill=tk.BOTH, expand=True)
    tree_scroll.config(command=tree.yview)

    # Sütun başlıkları
    tree.heading(header, text="Algılanan Cümle", anchor="center")
    tree.heading("Konu", text="Konu", anchor="center")

    # Sütun genişlikleri
    tree.column(header, anchor="center", width=400)
    tree.column("Konu", anchor="center", width=150)

    # Mikrofon üzerinden ses tanıma
    with sr.Microphone() as source:
        word_count = 0
        print("Ses tanıma başlatıldı...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)
                recognized_text = recognizer.recognize_google(audio, language="tr-TR")

                # Kelime sayısını artır
                word_count += len(recognized_text.split())
                word_count_label.config(text=f"Toplam Kelime Sayısı: {word_count}")

                # Konu belirleme
                topic = get_topics_from_keywords(recognized_text)

                # Uzun cümleleri böl ve Treeview'e ekle
                wrapped_lines = wrap_text(recognized_text, max_length=50)
                for idx, line in enumerate(wrapped_lines):
                    tree.insert("", "end", values=(line if idx == 0 else f"   {line}", topic if idx == 0 else ""))
            except sr.UnknownValueError:
                pass  # Anlaşılmayan sesleri yoksay
            except sr.RequestError:
                messagebox.showerror("Hata", "Google Speech API'ye bağlanılamadı!")
                break
