from globals import audio_data
def real_time_audio_histogram(indata , frames, time, status): # NOSONAR 
    """Ses verisinden histogram oluşturur ve günceller."""
    audio_data.extend(indata[:, 0])