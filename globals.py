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

def getRoot():
    return root

def setRoot(newRoot):
    global root
    root = newRoot


def getSpeakerModel():
    return speaker_model

def getLabelEncoder():
    return label_encoder

def getScaler():
    return scaler

def setSpeakerModel(newSpeakerModel):
    global speaker_model
    speaker_model = newSpeakerModel


def setLabelEncoder(newLabelEncoder):
    global label_encoder
    label_encoder = newLabelEncoder


def setScaler(newScaler):
    global scaler
    scaler = newScaler

def getStopFlag():
    return stop_flag

def setStopFlag(newStopFlag):
    global stop_flag
    stop_flag = newStopFlag
    
def getAudioAccuracy():
    return audio_accuracy

def setAudioAccuracy(newAudioAccuracy):
    global audio_accuracy
    audio_accuracy = newAudioAccuracy

def getAudioF1Score():
    return audio_f1_score

def setAudioF1Score(newAudioF1Score):
    global audio_f1_score
    audio_f1_score = newAudioF1Score

def getEmotionAccuracy():
    return emotion_accuracy

def setEmotionAccuracy(newEmotionAccuracy):
    global emotion_accuracy
    emotion_accuracy = newEmotionAccuracy
def getEmotionF1Score():
    return emotion_f1_score

def setEmotionF1Score(newEmotionF1Score):
    global emotion_f1_score
    emotion_f1_score = newEmotionF1Score