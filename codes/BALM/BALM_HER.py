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
    DatasetDict,
    Sequence,
    ClassLabel
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import pandas as pd
import torch
import numpy as np
import random
import os
import ipdb
from sklearn import metrics

def preprocess(batch): #这段在弄的事情是： seq在tokenizer之后长度会变，会根据最长的那个后面变多一定数量的token(就是padding) 前面也会加上起始的token 而label(1,0)的数量恒等于AA的数量 是不会变的 因此要给非token的label加上-100的标签
    sequence = []
    pos_ids = []
    for i in range(len(batch['VH'])):
        VH_seq = batch['VH'][i].replace(' ','')
        VL_seq = batch['VL'][i].replace(' ','')
        total_seq = VH_seq + VL_seq
        sequence.append(total_seq)

        pos = get_anarci_pos(total_seq)
        pos_ids.append(pos['position_ids'].numpy().tolist())

    tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)
    t_inputs = tokenizer(sequence, padding="max_length", truncation=True)
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

train_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/HER2/her2_train.csv")
val_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/HER2/her2_val.csv")
test_df = pd.read_csv("/home/hongnanqi/ALM_Benchmark/datasets/HER2/her2_test.csv")

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df[['VH', 'VL', 'labels']]),
    "valid": Dataset.from_pandas(val_df[['VH', 'VL', 'labels']]),
    "test": Dataset.from_pandas(test_df[['VH', 'VL', 'labels']])
})

dataset_tokenized = datasets.map(
    preprocess, 
    batched=True,
    batch_size=200,
    remove_columns=['VH', 'VL']
)


batch_size = 16
RUN_ID = "HER2-BALM"
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
    num_train_epochs=500,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='acc', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)
set_seed(SEED)

# We initialise a model using the weights from the pre-trained model
model = BALMForSequenceClassification.from_pretrained('/home/hongnanqi/ALM_Benchmark/models/BALM',num_labels=2)
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
    eval_dataset=dataset_tokenized['valid'], 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

trainer.train()

pred = trainer.predict(
    dataset_tokenized['test']
)

print(pred.metrics)
