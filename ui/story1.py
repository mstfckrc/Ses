from matplotlib import pyplot as plt1
import sounddevice as sd

from config import CHUNK, SAMPLE_RATE
from controller.audio.real_time import real_time_audio_histogram
from controller.histogram.update import stop_program, update_histogram
from globals import setStopFlag

def user_story_1():
    """Histogram oluşturulmasını sağlar."""
    setStopFlag(False)
    with sd.InputStream(callback=real_time_audio_histogram, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK):
        fig = plt1.figure()
        fig.canvas.mpl_connect('close_event', stop_program)
        update_histogram()