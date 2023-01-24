import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from train import  *
from save_to_json import *
from translate_resume import *


nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
bloc = []

example = translate(phrase = main())
ner_results = nlp(example)
for i in range(len(ner_results)):
    print(ner_results[i]["entity"],"->",ner_results[i]["word"])
for i in range(len(ner_results)):
    if ner_results[i]['entity'] == "B-LOC":
        bloc.append(ner_results[i]['word'])


def PER(data):
    list_per = []
    for i in range(len(data)-1):
         if data[i]['entity'] == "B-PER"or data[i]['entity'] == "I-PER":
             list_per.append(data[i]["word"])
    return list_per

x = PER(ner_results)
print(x)
i = 1
while i < len(x):

    if "##" not in x[i]:
        x.insert(x.index(x[i]), ",")
        i += 2
    else:
        x[i] = x[i].replace("##", "")
        i+=1

x = "".join(x).replace(",", " ")
0
country = input("Input city: ").lower()
otvet = [bloc[i] for i in range(len(bloc)) if bloc[i] == country]
if len(otvet) != 0:
    print(f"Matches was found at: {x}")
