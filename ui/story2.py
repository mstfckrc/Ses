import threading
from controller.audio.recognize import recognize_continuous


def user_story_2():
    """Ses tanıma işlemi başlatır ve tanınan metni ekrana yazdırır."""

    # Yeni bir thread (iş parçacığı) oluşturuyoruz, böylece ses tanıma işlemi ana thread'i engellemiyor
    threading.Thread(target=recognize_continuous, daemon=True).start()