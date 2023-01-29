from tokenize_and_setting_metrics import *
from transformers import TrainingArguments, Trainer

"""Обучаем нашу модель"""
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
