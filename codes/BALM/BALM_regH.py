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
from scipy.stats import pearsonr, spearmanr

def preprocess(batch):
    pos_ids = []
    sequence = batch['heavy']
    for i in range(len(sequence)):
        pos = get_anarci_pos(sequence[i])
        pos_ids.append(pos['position_ids'].numpy().tolist())

    tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)

    batch['labels'] = batch['affinity']
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

train_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/VH/VH_train.csv")
val_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/VH/VH_val.csv")
test_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/VH/VH_test.csv")

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df[['heavy', 'affinity']]),
    "validation": Dataset.from_pandas(val_df[['heavy', 'affinity']]),
    "test": Dataset.from_pandas(test_df[['heavy', 'affinity']])
})

dataset_tokenized = datasets.map(
    preprocess, 
    batched=True,
    batch_size=200,
    remove_columns=['heavy', 'affinity']
)

batch_size = 16
RUN_ID = "BALM_RegH"
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
    metric_for_best_model='rp_s', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)
set_seed(SEED)

# We initialise a model using the weights from the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BALMForSequenceClassification.from_pretrained('/home/hongnanqi/ALM_Benchmark/models/BALM',num_labels=1)
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
    train_dataset=dataset_tokenized['train'],
    eval_dataset=dataset_tokenized['validation'], 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

trainer.train()

pred = trainer.predict(
    dataset_tokenized['test']
)

print(pred.metrics)
