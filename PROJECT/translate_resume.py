from googletrans import Translator
from text_detection import *


def translate(phrase):
    translator = Translator()
    result = translator.translate(phrase, src = 'ru', dest = 'en')
    return result.text
print(translate(phrase = main()))
