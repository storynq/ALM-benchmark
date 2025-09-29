from modeling_balm import BALMForTokenClassification
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

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查当前使用的设备数量
print(f"Device count: {torch.cuda.device_count()}")

# 检查当前设备
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
else:
    print("Running on CPU")

def preprocess(batch): #这段在弄的事情是： seq在tokenizer之后长度会变，会根据最长的那个后面变多一定数量的token(就是padding) 前面也会加上起始的token 而label(1,0)的数量恒等于AA的数量 是不会变的 因此要给非token的label加上-100的标签
    sequence = batch['sequence']

    tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)
    t_inputs = tokenizer(sequence, truncation=True, padding="max_length")

    pos_ids = []
    for i in range(len(sequence)):
        pos = get_anarci_pos(sequence[i])
        pos_ids.append(pos['position_ids'].numpy().tolist())

    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['position_ids'] = pos_ids

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
    
    # The predictions are logits, so we apply softmax to get the probabilities. We only need
    # the probabilities of the paratope label, which is index 1 (according to the ClassLabel we made earlier),
    # or the last column from the output tensor
    prediction_pr = torch.softmax(torch.from_numpy(predictions), dim=2).detach().numpy()[:,:,-1]
    
    # We run an argmax to get the label
    predictions = np.argmax(predictions, axis=2)

    # Only compute on positions that are not labelled -100
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
            
    # flatten
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
    '/home/hongnanqi/ALM_Benchmark/datasets/paratope/sabdab_train_clustered.parquet'
)
val_df = pd.read_parquet(
    '/home/hongnanqi/ALM_Benchmark/datasets/paratope/sabdab_val_clustered.parquet'
)
test_df = pd.read_parquet(
    '/home/hongnanqi/ALM_Benchmark/datasets/paratope/sabdab_test_clustered.parquet'
)

ab_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[['sequence','paratope_labels']]),
    "validation": Dataset.from_pandas(val_df[['sequence','paratope_labels']]),
    "test": Dataset.from_pandas(test_df[['sequence','paratope_labels']])
})

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

train_temp = ab_dataset['train'].map(remove_columns = ['__index_level_0__'])
feature_set_copy = train_temp.features.copy()
feature_set_copy['paratope_labels'] = new_feature
ab_dataset_featurised = ab_dataset_featurised.cast(feature_set_copy)

ab_dataset_tokenized = ab_dataset_featurised.map(
    preprocess, 
    batched=True,
    batch_size=8,
    remove_columns=['sequence', 'paratope_labels']
)

label_list = paratope_class_label.names
batch_size = 32
RUN_ID = "paratope-prediction-BALM"
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
    metric_for_best_model='aupr', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)
print(ab_dataset_tokenized)
# We initialise a model using the weights from the pre-trained model
model = BALMForTokenClassification.from_pretrained('/home/hongnanqi/ALM_Benchmark/models/BALM', num_labels=2)
tokenizer = EsmTokenizer.from_pretrained("/home/hongnanqi/ALM_Benchmark/models/BALM/tokenizer", do_lower_case=False, model_max_length=168)

#best model for test
# model = RoFormerForTokenClassification.from_pretrained(r'E:\Antibody(1)\Antibody\无监督学习\code\paratope-prediction-task-antiberta2_0_old\checkpoint-1771')

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=ab_dataset_tokenized['train'],
    eval_dataset=ab_dataset_tokenized['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

print(model)

trainer.train()
pred = trainer.predict(
    ab_dataset_tokenized['test']
)

print(pred.metrics)
