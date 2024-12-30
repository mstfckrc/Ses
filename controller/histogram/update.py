from matplotlib import pyplot as plt
import numpy as np
from config import SAMPLE_RATE
from globals import get_stop_flag as getStopFlag , set_stop_flag as setStopFlag , audio_data

def update_histogram():
    """Zaman ve frekans domaini histogramlarını çizer."""
    setStopFlag(False)
    plt.ion()
    plt.show()
    while not getStopFlag():
        # Zaman domaini
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.plot(audio_data)
        plt.title("Zaman Domaini - Ses Dalga Formu")
        plt.xlabel("Zaman (örnekler)")
        plt.ylabel("Genlik")
        plt.xlim(0, len(audio_data))

        # Spektrogram
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.specgram(np.array(audio_data), Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, scale='dB', cmap='viridis')
        plt.title("Spektrogram - Frekans Domaini")
        plt.xlabel("Zaman (saniye)")
        plt.ylabel("Frekans (Hz)")

        plt.tight_layout()
        plt.pause(0.1)

def stop_program(event):
    """Histogram penceresi kapatıldığında döngüyü durdurur."""
    setStopFlag(True)
    plt.close()
