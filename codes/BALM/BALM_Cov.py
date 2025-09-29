from modeling_balm import BALMForSequenceClassification
from ba_position_embedding import get_anarci_pos
from transformers import (
    Trainer,
    TrainingArguments,
    EsmTokenizer,
    EarlyStoppingCallback
)

from datasets import (
    Dataset,
    concatenate_datasets,
    DatasetDict
)

import pandas as pd
import torch
import numpy as np
import random
import os
import ipdb
from sklearn import metrics

# 分类的字典
Covid_dict = {
    'SARS-CoV1': 0, 'SARS-CoV2_WT': 1, 'SARS-CoV2_Alpha': 2, 
    'SARS-CoV2_Beta': 3, 'SARS-CoV2_Gamma': 4, 'SARS-CoV2_Delta': 5, 
    'SARS-CoV2_Omicron-BA1': 6, 'SARS-CoV2_Omicron-BA2': 7, 
    'SARS-CoV2_Omicron-BA3': 8, 'SARS-CoV2_Omicron-XBB': 9
}


def preprocess(batch):
    labels = batch['Binds to']

    label_list = []
    sequence = []
    pos_ids = []
    for i in range(len(labels)):
        if labels[i] is None or not str(labels[i]).strip():
            onehot_code = np.zeros(10)
        else:
            lst = labels[i].split(',')
            result_list = [Covid_dict[item] for item in lst if item in Covid_dict]
            onehot_code= np.zeros(10)
            for j in result_list:
                onehot_code[j] = 1
        label_list.append(onehot_code)

        VH_seq = batch['VHorVHH'][i]
        VL_seq = batch['VL'][i]
        if batch['VL'][i] is None:
            total_seq = VH_seq 
        else:
            total_seq = VH_seq+VL_seq
        sequence.append(total_seq)

        pos = get_anarci_pos(total_seq)
        pos_ids.append(pos['position_ids'].numpy().tolist())

    tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)

    batch['labels'] = label_list
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['position_ids'] = pos_ids

    return batch

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


train_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/Covid/covid_train_X.csv")
val_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/Covid/covid_val_X.csv")
test_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/Covid/covid_test.csv")

train_df_drop = train_df.query("VL != 'ND'" or "VHorVHH != 'ND'")
val_df_drop = val_df.query("VL != 'ND'" or "VHorVHH != 'ND'")
test_df_drop = test_df.query("VL != 'ND'" or "VHorVHH != 'ND'")

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df_drop[['VHorVHH', 'VL', 'Binds to']]),
    "valid": Dataset.from_pandas(val_df_drop[['VHorVHH', 'VL', 'Binds to']]),
    "test": Dataset.from_pandas(test_df_drop[['VHorVHH', 'VL', 'Binds to']])
})

dataset_pred = datasets.map(
    preprocess,
    batched = True,
    batch_size= 32,
    remove_columns=['VHorVHH', 'VL', '__index_level_0__']
)

batch_size = 16
RUN_ID = "BALM_Covid"
SEED = 0
LR = 1e-6

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=3,
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=300,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='precision', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)
set_seed(SEED)

# We initialise a model using the weights from the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BALMForSequenceClassification.from_pretrained('/home/hongnanqi/ALM_Benchmark/models/BALM',num_labels=10)
tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)

total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('model:', model)
print('total_num:', total_num)
print('trainable_num:', trainable_num)

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset_pred['train'],
    eval_dataset=dataset_pred['valid'], 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

trainer.train()

pred = trainer.predict(dataset_pred['test'])

print(pred.metrics)
