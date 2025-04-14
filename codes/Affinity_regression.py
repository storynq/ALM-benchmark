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
from scipy.stats import pearsonr, spearmanr
trust_remote_code=True

def compute_metrics(p):
    predictions, labels = p   #pred:(eval_num, max_length,vocab_size) --> (100,170,30)  labels(eval_num, max_length) --> (100,170)
    labels = torch.tensor(labels)
    outputs = torch.tensor(predictions)
    outputs = torch.squeeze(outputs)
    rmse = np.sqrt(metrics.mean_squared_error(outputs, labels))
    r2 = metrics.r2_score(labels, outputs)   #r2 should be (y_true, y_pred)
    mae = metrics.mean_absolute_error(outputs, labels)
    rp = pearsonr(labels, outputs)  #rp有两个值 一个值一个p_value
    spearman = spearmanr(labels, outputs)


    return {
        "RMSE": rmse,
        "r2": r2,
        "MAE": mae,
        "rp_s": rp[0],
        "rp_p": rp[1],
        "sp_s": spearman[0],
        "sp_p": spearman[1]
    }

def set_seed(seed: int = 42):
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def preprocess(batch): 
    sequence = batch['heavy']
    tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150)  
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['labels'] = batch['affinity']
 
    return batch

data_file_H = pd.read_csv('./datasets/VH.csv')
data_file_L = pd.read_csv('./datasets/VL.csv')

datasets = Dataset.from_pandas(data_file_H[['heavy', 'affinity']])
dataset = datasets.shuffle(seed=42)
dataset_temp = dataset.train_test_split(test_size=0.2, shuffle=False)
train_temp = dataset_temp['train']
validset = dataset_temp['test']

dataset_final = train_temp.train_test_split(test_size=0.25, shuffle=False)
dataset_final['validation'] = validset

dataset_tokenized = dataset_final.map(
    preprocess, 
    batched=True,
    batch_size=200,
    remove_columns=['heavy', 'affinity']
)

batch_size = 32
RUN_ID = "RegH-antiberta"
SEED = 0
LR = 1e-5  

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=3,
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=200,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='RMSE', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

model = RobertaForSequenceClassification.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode', num_labels=1)
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