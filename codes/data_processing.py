# data_processing.py - 数据处理模块（修复版本）
from datasets import DatasetDict, Dataset, Sequence, ClassLabel
import pandas as pd
import copy
import torch
import numpy as np
import ipdb

class DataProcessor:
    def __init__(self):
        self.sequence_prepare = self._sequence_prepare
        self.helper_fn_infilling = self._helper_fn_infilling
    
    def _sequence_prepare(self, string, space):
        """Seqeunce preparation for Antiberta2, Antiberty and ReprogBert"""
        spaced = ' '.join(string[i:i+space] for i in range(0, len(string), space))
        return spaced
    
    def _helper_fn_infilling(self, src_ids, cdr):
        """infilling for cdr tasks"""
        src_ids = torch.tensor(src_ids)
        infill_loc_indices = []
        infill_mask = torch.zeros_like(src_ids).bool()
        for i, cdr_batch in enumerate(cdr):
            loc_list = []
            for j, charac in enumerate(cdr_batch):
                if str(charac) == "T":
                    loc_list.append(j + 1)
                    infill_mask[i, j+1] = True
            infill_loc_indices.append(loc_list)

        max_len = max([len(ele) for ele in infill_loc_indices])

        for idx in range(len(infill_loc_indices)):
            ele = infill_loc_indices[idx]
            ele = ele + [-1 for _ in range(max_len - len(ele))]
            infill_loc_indices[idx] = torch.LongTensor(ele)

        return torch.stack(infill_loc_indices), infill_mask
    
    def process_data(self, model_name, task_name, config, tokenizer, cdr_type=None):
        if task_name == 'paratope_prediction':
            return self._process_paratope_data(model_name, config, tokenizer)
        elif task_name.startswith('cdr'):
            return self._process_cdr_data(model_name, config, tokenizer, cdr_type)
        elif task_name == 'her2_prediction':
            return self._process_her2_data(model_name, config, tokenizer)
        elif task_name == 'covid_prediction':
            return self._process_covid_data(model_name, config, tokenizer)
        elif task_name == 'vh_prediction':
            return self._process_vh_data(model_name, config, tokenizer)
        elif task_name == 'vl_prediction':
            return self._process_vl_data(model_name, config, tokenizer)
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    def _process_paratope_data(self, model_name, config, tokenizer):

        train_df = pd.read_parquet(config['tasks']['paratope_prediction']['train_data'])
        val_df = pd.read_parquet(config['tasks']['paratope_prediction']['val_data'])
        test_df = pd.read_parquet(config['tasks']['paratope_prediction']['test_data'])
        
        ab_dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df[['sequence','paratope_labels']]),
            "validation": Dataset.from_pandas(val_df[['sequence','paratope_labels']]),
            "test": Dataset.from_pandas(test_df[['sequence','paratope_labels']])
        })
        
        paratope_class_label = ClassLabel(2, names=['N','P'])
        new_feature = Sequence(paratope_class_label)
        
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
        
        def preprocess(batch):
            sequence = batch['sequence']

            # special token for some models
            if model_name in ['antiberta2','antiberty','ReprogBert']:
                processed_sequence = []
                for seq in sequence:
                    processed_sequence.append(self.sequence_prepare(seq, 1))
                sequence = processed_sequence
            if model_name == 'Ablang2':
                processed_sequence = []
                for seq in sequence:
                    # Ablang2-heavy type
                    ab2_seq = f"<{seq}>|{''}"
                    processed_sequence.append(ab2_seq)
                sequence = processed_sequence


            if model_name == "Ablang":
                t_inputs = tokenizer(batch['sequence'], pad = True)
                batch['input_ids'] = t_inputs
            elif model_name == "Ablang2":
                t_inputs = tokenizer(sequence, pad=True, w_extra_tkns=False)
                batch['input_ids'] = t_inputs
            else:
                t_inputs = tokenizer(sequence, padding="max_length")
                batch['input_ids'] = t_inputs.input_ids
                batch['attention_mask'] = t_inputs.attention_mask
            
            # enumerate 
            labels_container = []
            for index, labels in enumerate(batch['paratope_labels']):
                tokenized_input_length = len(batch['input_ids'][index])
                paratope_label_length = len(batch['paratope_labels'][index])
                n_pads_with_eos = max(1, tokenized_input_length - paratope_label_length - 1)
                labels_padded = [-100] + labels + [-100] * n_pads_with_eos
                assert len(labels_padded) == len(batch['input_ids'][index]), \
                    f"Lengths don't align, {len(labels_padded)}, {len(batch['input_ids'][index])}, {len(labels)}"
                
                labels_container.append(labels_padded)
            
            batch['labels'] = labels_container
            
            return batch
        
        bz = 99999 if model_name in ['Ablang', 'Ablang2'] else 32
        ab_dataset_tokenized = ab_dataset_featurised.map(
            preprocess, 
            batched=True,
            batch_size=bz,
            remove_columns=['sequence', 'paratope_labels','__index_level_0__']
        )
        
        return ab_dataset_tokenized, paratope_class_label.names
    
    def _process_cdr_data(self, model_name, config, tokenizer, cdr_type):

        if cdr_type == "CDR1":
            replace_num = "1"
        elif cdr_type == "CDR2":
            replace_num = "2"
        elif cdr_type == "CDR3":
            replace_num = "3"
        else:
            raise ValueError(f"CDR_TYPE ERROR: {cdr_type}")
        
        task_config_key = f"cdr{cdr_type[-1]}_prediction" if f"cdr{cdr_type[-1]}_prediction" in config['tasks'] else "cdr_prediction"
        task_config = config['tasks'][task_config_key]
        
        train_data = pd.read_csv(task_config['train_data'])
        valid_data = pd.read_csv(task_config['val_data'])
        test_data = pd.read_csv(task_config['test_data'])

        cdr_column = cdr_type  # CDR1, CDR2 or CDR3
        
        train_drop = train_data.dropna(subset=[cdr_column])
        valid_drop = valid_data.dropna(subset=[cdr_column])
        test_drop = test_data.dropna(subset=[cdr_column])

        if model_name == 'antiberta':
            train_drop = train_drop[~(train_drop['Total_CDR'].str.rfind(replace_num) > 148)]
            valid_drop = valid_drop[~(valid_drop['Total_CDR'].str.rfind(replace_num) > 148)]
            test_drop = test_drop[~(test_drop['Total_CDR'].str.rfind(replace_num) > 148)]


        # proportion of training set
        # train_drop = train_drop.sample(frac=0.1, random_state=42)

        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_drop[['Sequence', 'Total_CDR', cdr_column, 'Type']]),
            "validation": Dataset.from_pandas(valid_drop[['Sequence', 'Total_CDR', cdr_column, 'Type']]),
            "test": Dataset.from_pandas(test_drop[['Sequence', 'Total_CDR', cdr_column, 'Type']])
        })
        
        def preprocess(batch, max_length = 256):
            sequence = batch['Sequence']
            cdr_total = batch['Total_CDR']
            
            cdr = [item.replace(replace_num, 'T') for item in cdr_total]
            t_cdr = []

            t_sequence = []
            for i in range(len(sequence)):
                if model_name in ['antiberta2','antiberty','ReprogBert']:
                    seq = self.sequence_prepare(sequence[i], 1)

                elif model_name == 'Ablang':
                    if len(sequence[i]) > 157:
                        seq = sequence[i][:157]
                        t_cdr.append(cdr[i][:157])
                    else:
                        seq = sequence[i]
                        t_cdr.append(cdr[i])

                elif model_name == 'Ablang2':
                    if batch['Type'][i] == 'VH':
                        seq = f"<{sequence[i]}>|{''}"
                    if batch['Type'][i] == 'VL':
                        seq =  f"{''}|<{sequence[i]}>"

                else:
                    seq = sequence[i]
                t_sequence.append(seq)
                
            
            if model_name == 'Ablang':
                t_inputs = tokenizer(t_sequence, pad = True)
                src_ids = copy.deepcopy(t_inputs)
                tgt_ids = copy.deepcopy(t_inputs)
                cdr = t_cdr
                tokenizer.mask_token_id = 23
            elif model_name == 'Ablang2':
                t_inputs = tokenizer(t_sequence,pad=True, w_extra_tkns=False)
                src_ids = copy.deepcopy(t_inputs)
                tgt_ids = copy.deepcopy(t_inputs)
            else:
                t_inputs = tokenizer(t_sequence, padding="max_length", truncation=True, max_length=max_length)
                src_ids = copy.deepcopy(t_inputs['input_ids'])
                tgt_ids = copy.deepcopy(t_inputs['input_ids'])

            infill_loc_indices, infill_mask = self.helper_fn_infilling(src_ids, cdr)

            for i in range(len(src_ids)):
                ids = src_ids[i]
                for j in infill_loc_indices[i]:
                    if j == -1:
                        continue
                    if model_name in ['Ablang', 'Ablang2']:
                        ids[j] = 23
                    else:
                        ids[j] = tokenizer.mask_token_id
                src_ids[i] = ids

            batch['input_ids'] = src_ids
            batch['labels'] = tgt_ids
            batch['infill_mask'] = infill_mask
            if model_name not in ['Ablang', 'Ablang2']:
                batch['attention_mask'] = t_inputs['attention_mask']

            return batch
        
        max_length = config['models'][model_name].get('max_length', 256)
        bz = 99999 if model_name == 'Ablang2' else 200
        dataset_tokenized = datasets.map(
            lambda batch: preprocess(batch, max_length=max_length),
            batched=True,
            batch_size=bz,
            remove_columns=['Sequence', 'Total_CDR', cdr_column]
        )


        return dataset_tokenized, None
    
    def _process_her2_data(self, model_name, config, tokenizer):

        train_df = pd.read_csv(config['tasks']['her2_prediction']['train_data'])
        val_df = pd.read_csv(config['tasks']['her2_prediction']['val_data'])
        test_df = pd.read_csv(config['tasks']['her2_prediction']['test_data'])
        
        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[['VH', 'VL', 'labels']]),
            "valid": Dataset.from_pandas(val_df[['VH', 'VL', 'labels']]),
            "test": Dataset.from_pandas(test_df[['VH', 'VL', 'labels']])
        })

        
        def preprocess(batch, max_length=229): 
            sequence = []

            for i in range(len(batch['VH'])):
                if model_name != 'Bert2DAb':
                    VH_seq = batch['VH'][i].replace(' ','')
                    VL_seq = batch['VL'][i].replace(' ','')
                    total_seq = VH_seq + VL_seq
                else:
                    VH_seq = batch['VH'][i]
                    VL_seq = batch['VL'][i]
                    total_seq = VH_seq + VL_seq

                if model_name in ['antiberta2','antiberty','ReprogBert']:
                    total_seq = self.sequence_prepare(total_seq, 1)

                if model_name == 'Ablang':
                    if len(total_seq) > 157:
                        total_seq = total_seq[:157]

                if model_name == 'Ablang2':
                    total_seq = f"<{VH_seq}>|<{VL_seq}>"
                
                sequence.append(total_seq)

            
            if model_name == 'Ablang':
                t_inputs = tokenizer(sequence)
                batch['input_ids'] = t_inputs
            elif model_name == 'Ablang2':
                t_inputs = tokenizer(sequence,pad=True, w_extra_tkns=False)
                batch['input_ids'] = t_inputs
            else:
                t_inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length = max_length)
                batch['input_ids'] = t_inputs.input_ids
                batch['attention_mask'] = t_inputs.attention_mask

            return batch
        

        max_length = config['models'][model_name].get('max_length', 229)
        # bz = 99999 if model_name == 'Ablang2' else 200   ## HER2 task has same VH&VL length
        dataset_tokenized = datasets.map(
            lambda batch: preprocess(batch, max_length=max_length),
            batched=True,
            batch_size=200,
            remove_columns=['VH', 'VL']
        )

        return dataset_tokenized, None
    
    def _process_covid_data(self, model_name, config, tokenizer):

        Covid_dict = {
            'SARS-CoV1': 0, 'SARS-CoV2_WT': 1, 'SARS-CoV2_Alpha': 2, 
            'SARS-CoV2_Beta': 3, 'SARS-CoV2_Gamma': 4, 'SARS-CoV2_Delta': 5, 
            'SARS-CoV2_Omicron-BA1': 6, 'SARS-CoV2_Omicron-BA2': 7, 
            'SARS-CoV2_Omicron-BA3': 8, 'SARS-CoV2_Omicron-XBB': 9
        }
        
        # Bert2DAb need unique dataset
        if model_name == 'Bert2DAb':
            train_df = pd.read_csv(config['tasks']['covid_prediction_Bert2DAb']['train_data'])
            val_df = pd.read_csv(config['tasks']['covid_prediction_Bert2DAb']['train_data'])
            test_df = pd.read_csv(config['tasks']['covid_prediction_Bert2DAb']['train_data'])

            datasets = DatasetDict({
                "train": Dataset.from_pandas(train_df[['VH_second', 'VL_second', 'Binds to']]),
                "valid": Dataset.from_pandas(val_df[['VH_second', 'VL_second', 'Binds to']]),
                "test": Dataset.from_pandas(test_df[['VH_second', 'VL_second', 'Binds to']])
            })

        else:
            if model_name == 'Ablang':
                train_df = pd.read_csv(config['tasks']['covid_prediction_X']['train_data'])
                val_df = pd.read_csv(config['tasks']['covid_prediction_X']['val_data'])
                test_df = pd.read_csv(config['tasks']['covid_prediction_X']['test_data'])
            else:
                train_df = pd.read_csv(config['tasks']['covid_prediction']['train_data'])
                val_df = pd.read_csv(config['tasks']['covid_prediction']['val_data'])
                test_df = pd.read_csv(config['tasks']['covid_prediction']['test_data'])

            train_df_drop = train_df.query("VL != 'ND'" or "VHorVHH != 'ND'")
            val_df_drop = val_df.query("VL != 'ND'" or "VHorVHH != 'ND'")
            test_df_drop = test_df.query("VL != 'ND'" or "VHorVHH != 'ND'")
            
            datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df_drop[['VHorVHH', 'VL', 'Binds to']]),
            "valid": Dataset.from_pandas(val_df_drop[['VHorVHH', 'VL', 'Binds to']]),
            "test": Dataset.from_pandas(test_df_drop[['VHorVHH', 'VL', 'Binds to']])
            })

        def preprocess(batch, max_length=256):
            labels = batch['Binds to']
            label_list = []
            sequence = []

            for i in range(len(labels)):
                if labels[i] is None or not str(labels[i]).strip():
                    onehot_code = np.zeros(10)
                else:
                    if model_name == 'Bert2DAb':
                        lst = labels[i].split(';')
                    else:
                        lst = labels[i].split(',')
                    result_list = [Covid_dict[item] for item in lst if item in Covid_dict]
                    onehot_code= np.zeros(10)
                    for j in result_list:
                        onehot_code[j] = 1
                label_list.append(onehot_code)

                if model_name == 'Bert2DAb':
                    VH_seq = batch['VH_second'][i]
                    VL_seq = batch['VL_second'][i]
                else:
                    VH_seq = batch['VHorVHH'][i]
                    VL_seq = batch['VL'][i]

                if VL_seq is None:
                    total_seq = VH_seq
                    if model_name == 'Ablang2':
                        total_seq = total_seq = f"<{VH_seq}>|{''}"
                else:
                    if model_name == 'Bert2DAb':
                        total_seq = VH_seq+ ' ' +VL_seq
                    elif model_name == 'Ablang2':
                        total_seq = f"<{VH_seq}>|<{VL_seq}>"
                    else:
                        total_seq = VH_seq+VL_seq
            
                if model_name in ['antiberta2','antiberty','ReprogBert']:
                    total_seq = self.sequence_prepare(total_seq, 1)
                
                if model_name == 'Ablang':
                    if len(total_seq) > 157:
                        total_seq = total_seq[:157]

                if model_name == 'Ablang2':
                    if len(total_seq) > 260:
                        total_seq = total_seq[:260]
            
                sequence.append(total_seq)

            if model_name == 'Ablang':
                t_inputs = tokenizer(sequence, pad=True)
                batch['labels'] = label_list
                batch['input_ids'] = t_inputs
                
            elif model_name == 'Ablang2':
                t_inputs = tokenizer(sequence,pad=True, w_extra_tkns=False)
                batch['labels'] = label_list
                batch['input_ids'] = t_inputs

            else:
                t_inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length = max_length)
                batch['labels'] = label_list
                batch['input_ids'] = t_inputs.input_ids
                batch['attention_mask'] = t_inputs.attention_mask

            return batch
        
        max_length = config['models'][model_name].get('max_length', 256)
        bz = 99999 if model_name in ['Ablang', 'Ablang2'] else 200   
        dataset_tokenized = datasets.map(
            lambda batch: preprocess(batch, max_length=max_length),
            batched=True,
            batch_size=bz,
        )

        return dataset_tokenized, None
    
    def _process_vh_data(self, model_name, config, tokenizer):

        # Bert2DAb need unique dataset
        if model_name == 'Bert2DAb':
            train_df = pd.read_csv(config['tasks']['vh_prediction_Bert2DAb']['train_data'])
            val_df = pd.read_csv(config['tasks']['vh_prediction_Bert2DAb']['train_data'])
            test_df = pd.read_csv(config['tasks']['vh_prediction_Bert2DAb']['train_data'])

            datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[['Sequence', 'affinity']]),
            "valid": Dataset.from_pandas(val_df[['Sequence', 'affinity']]),
            "test": Dataset.from_pandas(test_df[['Sequence', 'affinity']])
        })

        else:
            train_df = pd.read_csv(config['tasks']['vh_prediction']['train_data'])
            val_df = pd.read_csv(config['tasks']['vh_prediction']['val_data'])
            test_df = pd.read_csv(config['tasks']['vh_prediction']['test_data'])

            datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[['heavy', 'affinity']]),
            "valid": Dataset.from_pandas(val_df[['heavy', 'affinity']]),
            "test": Dataset.from_pandas(test_df[['heavy', 'affinity']])
        })

        def preprocess(batch, max_length=256):
            sequence = []
            if model_name == 'Bert2DAb':
                batch['heavy'] = batch['Sequence']
            for i in range(len(batch['heavy'])):
                total_seq= batch['heavy'][i]
                if model_name in ['antiberta2','antiberty','ReprogBert']:
                    seq = self.sequence_prepare(total_seq,1)
                elif model_name == 'Ablang2':
                    seq= f"<{total_seq}>|<>"
                else: 
                    seq = total_seq
                sequence.append(seq)
            

            if model_name == 'Ablang':
                t_inputs = tokenizer(sequence)
                batch['input_ids'] = t_inputs
                batch['labels'] = batch['affinity']

            elif model_name == 'Ablang2':
                t_inputs = tokenizer(sequence,pad=True, w_extra_tkns=False)
                batch['input_ids'] = t_inputs
                batch['labels'] = batch['affinity']
            
            else:
                t_inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length = max_length)
                batch['input_ids'] = t_inputs.input_ids
                batch['attention_mask'] = t_inputs.attention_mask
                batch['labels'] = batch['affinity']
        
            return batch

        max_length = config['models'][model_name].get('max_length', 256)
        bz = 99999 if model_name in ['Ablang', 'Ablang2'] else 200   
        dataset_tokenized = datasets.map(
            lambda batch: preprocess(batch, max_length=max_length),
            batched=True,
            batch_size=bz,
        )

        return dataset_tokenized, None
    
    def _process_vl_data(self, model_name, config, tokenizer):

        # Bert2DAb need unique dataset
        if model_name == 'Bert2DAb':
            train_df = pd.read_csv(config['tasks']['vl_prediction_Bert2DAb']['train_data'])
            val_df = pd.read_csv(config['tasks']['vl_prediction_Bert2DAb']['train_data'])
            test_df = pd.read_csv(config['tasks']['vl_prediction_Bert2DAb']['train_data'])

            datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[['Sequence', 'affinity']]),
            "valid": Dataset.from_pandas(val_df[['Sequence', 'affinity']]),
            "test": Dataset.from_pandas(test_df[['Sequence', 'affinity']])
        })

        else:
            train_df = pd.read_csv(config['tasks']['vl_prediction']['train_data'])
            val_df = pd.read_csv(config['tasks']['vl_prediction']['val_data'])
            test_df = pd.read_csv(config['tasks']['vl_prediction']['test_data'])

            datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[['light', 'affinity']]),
            "valid": Dataset.from_pandas(val_df[['light', 'affinity']]),
            "test": Dataset.from_pandas(test_df[['light', 'affinity']])
        })

        def preprocess(batch, max_length=256):
            sequence = []
            if model_name == 'Bert2DAb':
                batch['light'] = batch['Sequence']
            for i in range(len(batch['light'])):
                total_seq= batch['light'][i]
                if model_name in ['antiberta2','antiberty','ReprogBert']:
                    seq = self.sequence_prepare(total_seq,1)
                elif model_name == 'Ablang2':
                    seq= f"<>|<{total_seq}>"
                else: 
                    seq = total_seq
                sequence.append(seq)
            
            
            if model_name == 'Ablang':
                t_inputs = tokenizer(sequence)
                batch['input_ids'] = t_inputs
                batch['labels'] = batch['affinity']

            elif model_name == 'Ablang2':
                t_inputs = tokenizer(sequence,pad=True, w_extra_tkns=False)
                batch['input_ids'] = t_inputs
                batch['labels'] = batch['affinity']
            
            else:
                t_inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length = max_length)
                batch['input_ids'] = t_inputs.input_ids
                batch['attention_mask'] = t_inputs.attention_mask
                batch['labels'] = batch['affinity']
        
            return batch

        max_length = config['models'][model_name].get('max_length', 256)
        bz = 99999 if model_name in ['Ablang', 'Ablang2'] else 200   
        dataset_tokenized = datasets.map(
            lambda batch: preprocess(batch, max_length=max_length),
            batched=True,
            batch_size=bz,
        )

        return dataset_tokenized, None