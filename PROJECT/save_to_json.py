from train import *
from tokenize_and_setting_metrics import *
import json


model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")

id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}

config = json.load(open("ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model/config.json","w"))
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")
