from collections import deque
from sklearn.discriminant_analysis import StandardScaler

from config import SAMPLE_RATE


speaker_model = None
label_encoder = None
scaler = StandardScaler()  # Eğitimde kullanılan scaler, test sırasında da kullanılacak
audio_data = deque(maxlen=SAMPLE_RATE * 10)  # Histogram için 10 saniyelik veri
stop_flag = False  # Histogram döngüsü kontrolü
word_count = 0  # Toplam kelime sayısı global olarak tanımlandı
root = None
audio_accuracy = None
audio_f1_score = None
emotion_accuracy = None
emotion_f1_score = None

def get_root():
    return root

def set_root(new_root):
    global root
    root = new_root


def get_speaker_model():
    return speaker_model

def get_label_encoder():
    return label_encoder

def get_scaler():
    return scaler

def set_speaker_model(new_speaker_model):
    global speaker_model
    speaker_model = new_speaker_model


def set_label_encoder(new_label_encoder):
    global label_encoder
    label_encoder = new_label_encoder


def set_scaler(new_scaler):
    global scaler
    scaler = new_scaler

def get_stop_flag():
    return stop_flag

def set_stop_flag(new_stop_flag):
    global stop_flag
    stop_flag = new_stop_flag
    
def get_audio_accuracy():
    return audio_accuracy

def set_audio_accuracy(new_audio_accuracy):
    global audio_accuracy
    audio_accuracy = new_audio_accuracy

def get_audio_f1_score():
    return audio_f1_score

def set_audio_f1_score(new_audio_f1_score):
    global audio_f1_score
    audio_f1_score = new_audio_f1_score

def get_emotion_accuracy():
    return emotion_accuracy

def set_emotion_accuracy(new_emotion_accuracy):
    global emotion_accuracy
    emotion_accuracy = new_emotion_accuracy
def get_emotion_f1_score():
    return emotion_f1_score

def set_emotion_f1_score(new_emotion_f1_score):
    global emotion_f1_score
    emotion_f1_score = new_emotion_f1_score