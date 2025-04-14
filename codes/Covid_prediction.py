import pandas as pd
import torch
from datasets import (
    Dataset,
    concatenate_datasets,
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

data_file1 = pd.read_csv('./datasets/SARS-CoV1.csv')
data_file2 = pd.read_csv('./datasets/SARS-CoV2.csv')
data_file_drop = data_file1.query("VL != 'ND'" or "VHorVHH != 'ND'")
data_file_drop2 = data_file2.query("VL != 'ND'" or "VHorVHH != 'ND'")
dataset1 = Dataset.from_pandas(data_file_drop[['VHorVHH', 'VL', 'Binds to']]) 
dataset2 = Dataset.from_pandas(data_file_drop2[['VHorVHH', 'VL', 'Binds to']]) 

Covid_dict = {'SARS-CoV1':1, 'SARS-CoV2_WT':2, 'SARS-CoV2_Alpha':3, 'SARS-CoV2_Beta':4, 'SARS-CoV2_Gamma':5, 'SARS-CoV2_Delta':6, 'SARS-CoV2_Omicron-BA1':7, 'SARS-CoV2_Omicron-BA2':8, 'SARS-CoV2_Omicron-BA3':9, 'SARS-CoV2_Omicron-XBB':10}

def sequence_prepare(string, space):
    spaced = ' '.join(string[i:i+space] for i in range(0,len(string),space))
    return spaced

def preprocess(batch):
    labels = batch['Binds to']

    label_list = []
    sequence = []
    for i in range(len(labels)):
        lst = labels[i].split(';')
        result_list = [Covid_dict[item] for item in lst if item in Covid_dict]
        onehot_code= np.zeros(10)
        for j in result_list:
            onehot_code[j-1] = 1
        label_list.append(onehot_code)

        VH_seq = batch['VHorVHH'][i]
        VL_seq = batch['VL'][i]
        if batch['VL'][i] is None:
            total_seq = VH_seq 
        else:
            total_seq = VH_seq+VL_seq
        seq = sequence_prepare(total_seq, 1)
        sequence.append(seq)

    tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150)  #这里看起来total_num大部分在230上下 实际上设定成260可能就够了？
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)

    batch['labels'] = label_list
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask

    return batch

def compute_metrics(p):

    predictions, labels = p 
    labels = torch.tensor(labels)  
    y_pred = torch.tensor((predictions > 0.5).astype(int))

    f1 = metrics.f1_score(y_pred, labels,average='micro')
    precision = metrics.precision_score(y_pred, labels,average='micro')
    recall = metrics.recall_score(y_pred, labels,average='micro')
    total_acc = metrics.accuracy_score(y_pred.view(-1), labels.view(-1))


    return {
        "acc": total_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
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

dataset_pred1 = dataset1.map(
    preprocess,
    batched = True,
    batch_size= 32
)

dataset_pred2 = dataset2.map(
    preprocess,
    batched = True,
    batch_size= 32
)

combined_dataset = concatenate_datasets([dataset_pred1, dataset_pred2])

df1 = dataset_pred1.to_pandas()
df2 = dataset_pred2.to_pandas()
combined_df = pd.concat([df1, df2], ignore_index=True)
df_deduplicated = combined_df.drop_duplicates(subset = ['VHorVHH','VL'])
dataset_deduplicated = Dataset.from_pandas(df_deduplicated[['VHorVHH','VL','labels','input_ids','attention_mask']])
dataset_deduplicated = dataset_deduplicated.remove_columns(['VHorVHH', 'VL', '__index_level_0__'])

dataset = dataset_deduplicated.shuffle(seed=42)

total_samples = len(dataset)
train_samples = int(0.6 * total_samples)
valid_samples = int(0.2 * total_samples)
test_samples = total_samples - train_samples - valid_samples

train_dataset = dataset.select(indices = list(range(train_samples)))
valid_dataset = dataset.select(indices = list(range(train_samples, train_samples + valid_samples)))
test_dataset = dataset.select(indices = list(range(train_samples + valid_samples, total_samples)))

batch_size = 32
RUN_ID = "CovidDab-antiberta"
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
    num_train_epochs=200,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='precision', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

model = RobertaForSequenceClassification.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode',num_labels=10)
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

pred = trainer.predict(test_dataset)

print(pred.metrics)