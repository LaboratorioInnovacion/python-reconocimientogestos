import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)  # velocidad
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()
