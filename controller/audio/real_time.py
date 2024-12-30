from globals import audio_data

## noinspection PyArgumentEqualDefault
def real_time_audio_histogram(indata , frames, time, status):
    """Ses verisinden histogram oluşturur ve günceller."""
    audio_data.extend(indata[:, 0])