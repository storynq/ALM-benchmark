import pandas as pd
import torch
from datasets import (
    Dataset,
)
from transformers import (
    Trainer,
    TrainingArguments,
)

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn import metrics
import numpy as np
import random
import os
import ipdb
trust_remote_code=True

def sequence_prepare(string, space):
    spaced = ' '.join(string[i:i+space] for i in range(0,len(string),space))
    return spaced

def preprocess(batch):
    sequence = []

    for i in range(len(batch['VH'])):
        VH_seq = batch['VH'][i].replace(' ','')
        VL_seq = batch['VL'][i].replace(' ','')
        total_seq = VH_seq + VL_seq
        sequence.append(VH_seq)

    tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150)
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask

    return batch

def compute_metrics(p):

    predictions, labels = p   
    labels = torch.tensor(labels)
    outputs = torch.tensor(predictions)

    pred_ids = torch.argmax(outputs, dim=-1)
    acc = metrics.accuracy_score(pred_ids, labels)
    f1 = metrics.f1_score(pred_ids, labels)
    precision = metrics.precision_score(pred_ids, labels)
    recall = metrics.recall_score(pred_ids, labels)


    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


data_file = pd.read_csv('./datasets/Her2_dataset.csv')

dataset = Dataset.from_pandas(data_file[['VH', 'VL', 'labels']]) #In this dataset, each VH has 120 AAs and each VL has 107 aas. 

dataset_temp = dataset.train_test_split(test_size=0.2, shuffle=False)
train_temp = dataset_temp['train']
validset = dataset_temp['test']

dataset_final = train_temp.train_test_split(test_size=0.25, shuffle=False)
dataset_final['validation'] = validset

dataset_tokenized = dataset_final.map(
    preprocess, 
    batched=True,
    batch_size=200,
    remove_columns=['VH', 'VL']
)

batch_size = 32
RUN_ID = "HER2-antiberta"
SEED = 0
LR = 1e-6

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=5,
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=500,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='acc', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

model = RobertaForSequenceClassification.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode',num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150) 

total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('model:', model)
print('total_num:', total_num)
print('trainable_num:', trainable_num)

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset_tokenized['train'],
    eval_dataset=dataset_tokenized['validation'], 
    compute_metrics=compute_metrics
)

trainer.train()

pred = trainer.predict(
    dataset_tokenized['test']
)

print(pred.metrics)