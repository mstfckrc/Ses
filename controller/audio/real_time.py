from globals import audio_data

# NOSONAR: Kullanılmayan parametreler callback uyumluluğu için gereklidir.
def real_time_audio_histogram(indata , frames, time, status):
    """Ses verisinden histogram oluşturur ve günceller."""
    audio_data.extend(indata[:, 0])