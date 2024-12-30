from matplotlib import pyplot as plt2
import numpy as np
from config import SAMPLE_RATE
from globals import getStopFlag, setStopFlag, audio_data

def update_histogram():
    """Zaman ve frekans domaini histogramlarını çizer."""
    setStopFlag(False)
    plt2.ion()
    plt2.show()
    while not getStopFlag():
        # Zaman domaini
        plt2.subplot(2, 1, 1)
        plt2.cla()
        plt2.plot(audio_data)
        plt2.title("Zaman Domaini - Ses Dalga Formu")
        plt2.xlabel("Zaman (örnekler)")
        plt2.ylabel("Genlik")
        plt2.xlim(0, len(audio_data))

        # Spektrogram
        plt2.subplot(2, 1, 2)
        plt2.cla()
        plt2.specgram(np.array(audio_data), Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, scale='dB', cmap='viridis')
        plt2.title("Spektrogram - Frekans Domaini")
        plt2.xlabel("Zaman (saniye)")
        plt2.ylabel("Frekans (Hz)")

        plt2.tight_layout()
        plt2.pause(0.1)

def stop_program(event):
    """Histogram penceresi kapatıldığında döngüyü durdurur."""
    setStopFlag(True)
    plt2.close()
