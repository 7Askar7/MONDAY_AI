import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer


conll2003 = datasets.load_dataset("conll2003")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

example_token = conll2003["train"][0]
tokenizer_input = tokenizer(example_token["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenizer_input['input_ids'])
word_ids = tokenizer_input.word_ids()



def tokenize_and_align_labels(examples, label_all_tokens=True):
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

for token, label in zip(tokenizer.convert_ids_to_tokens(q["input_ids"][0]),q["labels"][0]):
    print(f"{token:_<40} {label}")

tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched = True)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels = 9)

args = TrainingArguments("test-ner",
                         evaluation_strategy='epoch',
                         learning_rate= 2e-5,
                         per_device_train_batch_size=16,
                         per_device_eval_batch_size=16,
                         num_train_epochs=3,
                         weight_decay=0.01,
                         )
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = datasets.load_metric('seqeval')
example = conll2003['train'][0]
label_list = conll2003["train"].features["ner_tags"].feature.names

labels = [label_list[i] for i in example["ner_tags"]]
metric.compute(predictions=[labels], references=[labels])

def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
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
