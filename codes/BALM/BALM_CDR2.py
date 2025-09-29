from modeling_balm import BALMForMaskedLM
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
import copy

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
    cdr_total = batch['Total_CDR'] # 用了对应的dataset_tag CDR变成了 0000000TTTTTTT000000022222200000033333000000这种
    cdr = [item.replace('2', 'T') for item in cdr_total]

    pos_ids = []
    for i in range(len(sequence)):
        pos = get_anarci_pos(sequence[i])
        pos_ids.append(pos['position_ids'].numpy().tolist())

    tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)

    t_inputs = tokenizer(sequence, truncation=True, padding="max_length")

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
    batch['infill_mask'] = infill_mask.numpy().tolist()
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
    outputs = torch.tensor(predictions[0])
    infill_mask = predictions[1]

    pred_ids = torch.argmax(outputs, dim=-1)
    num_elem = infill_mask.sum().item()
    infill_mask = torch.tensor(infill_mask)

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

train_data = pd.read_csv('/home/hongnanqi/ALM_Benchmark/datasets/CDR/cdr_train.csv')
valid_data = pd.read_csv('/home/hongnanqi/ALM_Benchmark/datasets/CDR/cdr_val.csv')
test_data = pd.read_csv('/home/hongnanqi/ALM_Benchmark/datasets/CDR/cdr_test.csv')

train_drop = train_data.dropna(subset = ['CDR2'])
valid_drop = valid_data.dropna(subset = ['CDR2'])
test_drop = test_data.dropna(subset = ['CDR2'])

train_filtered = train_drop[~(train_drop['Total_CDR'].str.rfind('2') > 166)]
valid_filtered = valid_drop[~(valid_drop['Total_CDR'].str.rfind('2') > 166)]
test_filtered = test_drop[~(test_drop['Total_CDR'].str.rfind('2') > 166)]

datasets = DatasetDict({
    "train": Dataset.from_pandas(train_filtered[['Sequence', 'Total_CDR', 'CDR2']]),
    "valid": Dataset.from_pandas(valid_filtered[['Sequence', 'Total_CDR', 'CDR2']]),
    "test": Dataset.from_pandas(test_filtered[['Sequence', 'Total_CDR', 'CDR2']])
})


dataset_tokenized = datasets.map(
    preprocess, 
    batched=True,
    batch_size=32,
    remove_columns=['Sequence', 'Total_CDR', 'CDR2']
)


batch_size = 16
RUN_ID = "CDR2-BALM"
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
    num_train_epochs=800,
    warmup_ratio=0, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='AAR', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

# We initialise a model using the weights from the pre-trained model
model = BALMForMaskedLM.from_pretrained('/home/hongnanqi/ALM_Benchmark/models/BALM')
tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)

#best model for test
# model = RoFormerForTokenClassification.from_pretrained(r'E:\Antibody(1)\Antibody\无监督学习\code\paratope-prediction-task-antiberta2_0_old\checkpoint-1771')

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
