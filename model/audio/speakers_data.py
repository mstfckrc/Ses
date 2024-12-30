import os

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