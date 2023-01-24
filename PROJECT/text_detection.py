import tika
from tika import parser



def main():
    l = []
    tika.initVM()
    path = str(input("Введите путь до вашего резюме: "))
    parsed_pdf = parser.from_file(path)
    print(parsed_pdf)

    for my_key, my_value in parsed_pdf["metadata"].items():
        print(f'{my_key}')
        print(f'\t{my_value}\n')

    my_content = parsed_pdf['content']
    print(type(my_content))
    l.append(my_content)

    l2 = []
    t1 = map(lambda s: s.strip('\n'), l)
    l2.append(*t1)
    rep = []
    new = []
    for x in l2:
        rep.append(x.replace("\n", " "))

    for j in rep:
        new.append(j.replace("\xa0", " "))
    x = "".join(new)
    return x


