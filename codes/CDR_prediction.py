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
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments
)

import numpy as np
import random
import os
import copy
import ipdb
trust_remote_code=True

def helper_fn_infilling(src_ids, cdr):
    src_ids = torch.tensor(src_ids)
    infill_loc_indices = []
    infill_mask = torch.zeros_like(src_ids).bool()
    for i, cdr_batch in enumerate(cdr):
        loc_list = []
        for j, charac in enumerate(cdr_batch):
            if str(charac) == "T":
                loc_list.append(j + 1)
                infill_mask[i,j+1] = True
        infill_loc_indices.append(loc_list)

    max_len = max([len(ele) for ele in infill_loc_indices])

    for idx in range(len(infill_loc_indices)):
        ele = infill_loc_indices[idx]
        ele = ele + [-1 for _ in range(max_len - len(ele))]
        infill_loc_indices[idx] = torch.LongTensor(ele)

    return torch.stack(infill_loc_indices), infill_mask

def preprocess(batch):
    sequence = batch['Sequence']
    cdr_total = batch['Total_CDR'] 
    cdr = [item.replace('1', 'T') for item in cdr_total]

    tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150) 

    t_inputs = tokenizer(sequence, padding="max_length",truncation=True)

    src_ids = copy.deepcopy(t_inputs['input_ids'])
    tgt_ids = copy.deepcopy(t_inputs['input_ids'])

    infill_loc_indices, infill_mask = helper_fn_infilling(src_ids, cdr)

    for i in range(len(src_ids)):
        ids = src_ids[i]
        for j in infill_loc_indices[i]:
            if j == -1:
                continue
            ids[j] = tokenizer.mask_token_id
        src_ids[i] = ids

    batch['input_ids'] = src_ids
    batch['labels'] = tgt_ids
    batch['attention_mask'] = t_inputs['attention_mask']
    batch['infill_mask'] = infill_mask

    return batch

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_metrics(p):
    predictions, labels = p   #pred:(eval_num, max_length,vocab_size) --> (100,170,30)  labels(eval_num, max_length) --> (100,170)
    labels = torch.tensor(labels)
    outputs = torch.tensor(predictions[0])
    infill_mask = predictions[1]

    pred_ids = torch.argmax(outputs, dim=-1)
    num_elem = infill_mask.sum().item()

    matches = labels[infill_mask] == pred_ids[infill_mask]
    succ_matches = matches.sum().item()
    total_matches = len(matches)

    loss_fnc = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    L = loss_fnc(outputs[infill_mask],labels[infill_mask])
    loss_elem = L * num_elem
    accuracy = 100 * (succ_matches / total_matches)

    
    return {
        "AAR": accuracy,
        "loss_elem": loss_elem
    }

dataset_tag = pd.read_excel('./datasets/CDR_prediction.xlsx')
data_drop = dataset_tag.dropna(subset = ['CDR1']) #delete CDR1 with NAN
condition = data_drop['Total_CDR'].str.rfind('1') > 148 
data_filtered = data_drop[~condition]
datasets = Dataset.from_pandas(data_filtered[['Sequence', 'Total_CDR', 'CDR1']]) 

dataset_tokenized = datasets.map(
    preprocess, 
    batched=True,
    batch_size=32,
    remove_columns=['Sequence', 'Total_CDR', 'CDR1', '__index_level_0__']
)

dataset = dataset_tokenized.shuffle(seed=42)
total_samples = len(dataset)
train_samples = int(0.6 * total_samples)
valid_samples = int(0.2 * total_samples)
test_samples = total_samples - train_samples - valid_samples

train_dataset = dataset.select(indices = list(range(train_samples)))
valid_dataset = dataset.select(indices = list(range(train_samples, train_samples + valid_samples)))
test_dataset = dataset.select(indices = list(range(train_samples + valid_samples, total_samples)))

print('Dataset:', len(train_dataset), len(valid_dataset), len(test_dataset))
batch_size = 32
RUN_ID = "newsabdab1-antiberta"
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
    num_train_epochs=800,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='AAR', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

model = RobertaForMaskedLM.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode',num_labels=2)
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
    train_dataset=train_dataset,
    eval_dataset=valid_dataset, 
    compute_metrics=compute_metrics
)

trainer.train()

pred = trainer.predict(
    test_dataset
)

print(pred.metrics)