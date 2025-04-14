from transformers import (
    Trainer,
    TrainingArguments,
    RobertaTokenizer,
    RobertaForTokenClassification
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
#paratope prediction for antiberta2
#sequence need to be like 'A A R A' other than 'AARA' for tokenizer, using sequence-prepare to add ' ' between amino acids.


def preprocess(batch): 
    sequence = batch['sequence']

    tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150)
    t_inputs = tokenizer(sequence, padding="max_length")
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask

    
    # enumerate 
    labels_container = []
    for index, labels in enumerate(batch['paratope_labels']):
        tokenized_input_length = len(batch['input_ids'][index])
        paratope_label_length  = len(batch['paratope_labels'][index])
        n_pads_with_eos = max(1, tokenized_input_length - paratope_label_length - 1)
        labels_padded = [-100] + labels + [-100] * n_pads_with_eos  #这是一个list合并 就是把本来的labels list 前面加一个-100， 后面加n_pads_with_eos个-100    
        assert len(labels_padded) == len(batch['input_ids'][index]), \
        f"Lengths don't align, {len(labels_padded)}, {len(batch['input_ids'][index])}, {len(labels)}"
        
        labels_container.append(labels_padded)
    
    # We create a new column called `labels`, which is recognised by the HF trainer object
    batch['labels'] = labels_container
    
    for i,v in enumerate(batch['labels']):
        assert len(batch['input_ids'][i]) == len(v) == len(batch['attention_mask'][i])
    
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
    """
    A callback added to the trainer so that we calculate various metrics via sklearn
    """
    predictions, labels = p
    
    prediction_pr = torch.softmax(torch.from_numpy(predictions), dim=2).detach().numpy()[:,:,-1]
    
    predictions = np.argmax(predictions, axis=2)

    preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labs = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    probs = [ 
        [prediction_pr[i][pos] for (pr, (pos, l)) in zip(prediction, enumerate(label)) if l!=-100]
         for i, (prediction, label) in enumerate(zip(predictions, labels)) 
    ] 
            
    preds = sum(preds, [])
    labs = sum(labs, [])
    probs = sum(probs,[])
    
    return {
        "precision": precision_score(labs, preds, pos_label="P"),
        "recall": recall_score(labs, preds, pos_label="P"),
        "f1": f1_score(labs, preds, pos_label="P"),
        "auc": roc_auc_score(labs, probs),
        "aupr": average_precision_score(labs, probs, pos_label="P"),
        "mcc": matthews_corrcoef(labs, preds),
    }

train_df = pd.read_parquet(
    './datasets/Paratope/sabdab_train.parquet'
)
val_df = pd.read_parquet(
    './datasets/Paratope/sabdab_train.parquet'
)
test_df = pd.read_parquet(
    './datasets/Paratope/sabdab_train.parquet'
)

train_old = Dataset.from_pandas(train_df[['sequence','paratope_labels']])
valid_old  = Dataset.from_pandas(val_df[['sequence','paratope_labels']])
test_old  = Dataset.from_pandas(test_df[['sequence','paratope_labels']])

sequence = valid_old['sequence'] + test_old['sequence']
lables = valid_old['paratope_labels']+ test_old['paratope_labels']
index_level = valid_old['__index_level_0__']+ test_old['__index_level_0__']
my_dict = {'sequence':sequence, 'paratope_labels':lables, '__index_level_0__': index_level}
valid_dataset = Dataset.from_dict(my_dict)

ab_dataset = train_old.train_test_split(test_size = 0.25, shuffle = False)
ab_dataset['validation'] = valid_dataset

paratope_class_label = ClassLabel(2, names=['N','P'])
new_feature = Sequence(
    paratope_class_label
)
ab_dataset_featurised = ab_dataset.map(
    lambda seq, labels: {
        "sequence": seq,
        "paratope_labels": [paratope_class_label.str2int(sample) for sample in labels]
    }, 
    input_columns=["sequence", "paratope_labels"], batched=True
)

feature_set_copy = ab_dataset['train'].features.copy()
feature_set_copy['paratope_labels'] = new_feature
ab_dataset_featurised = ab_dataset_featurised.cast(feature_set_copy)

ab_dataset_tokenized = ab_dataset_featurised.map(
    preprocess, 
    batched=True,
    batch_size=8,
    remove_columns=['sequence', 'paratope_labels','__index_level_0__']
)

label_list = paratope_class_label.names

batch_size = 32
RUN_ID = "paratope-prediction-task-antiberta"
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
    metric_for_best_model='aupr', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)
print(ab_dataset_tokenized)
model = RobertaForTokenClassification.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode', num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained('/home/nanqi/antibody/test_code/antiberta_mode/tokenizer', max_len=150) 

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=ab_dataset_tokenized['train'],
    eval_dataset=ab_dataset_tokenized['validation'],
    compute_metrics=compute_metrics
)

print(model)

trainer.train()
pred = trainer.predict(
    ab_dataset_tokenized['test']
)

print(pred.metrics)