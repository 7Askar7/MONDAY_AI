from save_to_json import *
from translate_resume import translate
from text_detection import detection

if __name__ == '__main__':

    example = translate(phrase=detection())
    ner_results = nlp(example)
    bloc = []

    for i in range(len(ner_results)):
        if ner_results[i]['entity'] == "B-LOC":
            bloc.append(ner_results[i]['word'])


    def PER(data):
        list_per = []
        for i in range(len(data) - 1):
            if data[i]['entity'] == "B-PER" or data[i]['entity'] == "I-PER":
                list_per.append(data[i]["word"])
        return list_per


    x = PER(ner_results)

    i = 1
    while i < len(x):

        if "##" not in x[i]:
            x.insert(x.index(x[i]), ",")
            i += 2
        else:
            x[i] = x[i].replace("##", "")
            i += 1

    x = "".join(x).replace(",", " ")

    country = input("Input city: ").lower()
    otvet = [bloc[i] for i in range(len(bloc)) if bloc[i] == country]
    if len(otvet) != 0:
        print(f"Matches was found at: {x}")
