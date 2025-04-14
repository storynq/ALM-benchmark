from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

text_datasets = {
    "train": ['./datasets/train-slice.txt'],
    "eval": ['./datasets/val-slice.txt'],
    "test": ['./datasets/test-slice.txt']
}

dataset = load_dataset("text", data_files=text_datasets)
tokenized_dataset = dataset.map(
    lambda z: tokenizer(
        z["text"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    batched=True,
    num_proc=1,
    remove_columns=["text"],
)

antiberta_config = {
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}

model_config = RobertaConfig(
    vocab_size=antiberta_config.get("vocab_size"),
    hidden_size=antiberta_config.get("hidden_size"),
    max_position_embeddings=antiberta_config.get("max_position_embeddings"),
    num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
    num_attention_heads=antiberta_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)

args = TrainingArguments(
    output_dir="test",
    overwrite_output_dir=True,
    per_device_train_batch_size=antiberta_config.get("batch_size", 32),
    per_device_eval_batch_size=antiberta_config.get("batch_size", 32),
    max_steps=100000,
    save_steps=5000,
    logging_steps=1000,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 1),
    fp16=True,
    evaluation_strategy="steps",
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)

trainer.train()

save_dir = '/home/nanqi/antibody/test_code/pre_train/model_save'
trainer.save_model(save_dir)
out = trainer.predict(tokenized_dataset['test'])
print(out['predictions'])