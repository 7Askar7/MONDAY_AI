from googletrans import Translator
from text_detection import *


def translate(phrase):
    """Переводим текст, который считали с рзеюме"""
    translator = Translator()
    result = translator.translate(phrase, src = 'ru', dest = 'en')
    return result.text

