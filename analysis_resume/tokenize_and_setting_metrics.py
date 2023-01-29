import datasets
import numpy as np
from transformers import BertTokenizerFast, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import evaluate

EVAL_STRATEGY = "epoch"
LER_RATE = 2e-5
TRAIN_BATCH = 16
TEST_BATCH = 16
TRAIN_EPOCHS = 3
WEIGHT_DEC = 0.01

conll2003 = datasets.load_dataset("conll2003")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

example_token = conll2003["train"][0]
tokenizer_input = tokenizer(example_token["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenizer_input['input_ids'])
word_ids = tokenizer_input.word_ids()


def tokenize_and_align_labels(examples, label_all_tokens=True):
    """1) Устанавливает -100 в качестве метки для этих специальных лексем и подслов,
    которые мы хотим замаскировать во время обучения
    2) Маскирует представления подслов после первого подслова"""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


q = tokenize_and_align_labels(conll2003['train'][4:5])

for token, label in zip(tokenizer.convert_ids_to_tokens(q["input_ids"][0]), q["labels"][0]):
    print(f"{token:_<40} {label}")

tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)

args = TrainingArguments("test-ner",
                         evaluation_strategy=EVAL_STRATEGY,
                         learning_rate=LER_RATE,
                         per_device_train_batch_size=TRAIN_BATCH,
                         per_device_eval_batch_size=TRAIN_BATCH,
                         num_train_epochs=TRAIN_EPOCHS,
                         weight_decay=WEIGHT_DEC,
                         )
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load('seqeval')
example = conll2003['train'][0]
label_list = conll2003["train"].features["ner_tags"].feature.names

labels = [label_list[i] for i in example["ner_tags"]]
metric.compute(predictions=[labels], references=[labels])


def compute_metrics(eval_preds):
    """1) Эта функция compute_metrics() сначала берет argmax логитов, чтобы преобразовать их в предсказания.
    2) Затем нам нужно преобразовать метки и предсказания из целых чисел в строки. Мы удаляем все значения,
    для которых метка равна -100, а затем передаем результаты в метод metric.compute():"""
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
