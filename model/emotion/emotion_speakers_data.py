import os

def load_emotions_data(data_dir="data/emotion_records"):
    """Klasörlerden duygu verilerini yükler."""
    emotions = {}
    if not os.path.exists(data_dir):
        print(f"'{data_dir}' klasörü bulunamadı!")
        return emotions

    for speaker in os.listdir(data_dir):
        speaker_dir = os.path.join(data_dir, speaker)
        if os.path.isdir(speaker_dir):
            emotions[speaker] = [
                (os.path.join(speaker_dir, f), f.split("_")[-1].replace(".wav", ""))
                for f in os.listdir(speaker_dir)
                if f.endswith(".wav")
            ]
    return emotions
