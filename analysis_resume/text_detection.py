import tika
from tika import parser


def detection():
    """Считываем pdf резюме и перевеодим в текст"""
    list_for_text = []
    tika.initVM()
    path_to_resume = str(input("Введите путь до резюме: "))
    parsed_pdf = parser.from_file(path_to_resume)

    for my_key, my_value in parsed_pdf["metadata"].items():
          print(f'{my_key}')
          print(f'\t{my_value}\n')

    my_content = parsed_pdf['content']
    list_for_text.append(my_content)

    text = []
    txt = map(lambda s: s.strip("\n"), list_for_text)
    text.append(*txt)

    list_for_replace_enter = []
    list_for_replace_symbol = []

    for x in text:
        list_for_replace_enter.append(x.replace("\n", " "))

    for j in list_for_replace_enter:
        list_for_replace_symbol.append(j.replace("\xa0", " "))
    itog_text = "".join(list_for_replace_symbol)
    return itog_text

