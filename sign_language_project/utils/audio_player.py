# utils/audio_player.py
import pyttsx3, threading
engine = pyttsx3.init()
engine.setProperty('rate', 150)

_lock = threading.Lock()
def speak(text):
    # non-blocking speak to avoid freezing the main loop
    def _run(t):
        with _lock:
            engine.say(t)
            engine.runAndWait()
    import threading
    threading.Thread(target=_run, args=(text,), daemon=True).start()
