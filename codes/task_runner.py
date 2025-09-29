# task_runners.py 
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn import metrics
import torch
import numpy as np
from scipy.stats import pearsonr
import ipdb

class TaskRunner:
    def run_task(self, model_name, task_name, model, tokenizer, dataset, label_list, model_config, config, args):
        if task_name == "paratope_prediction":
            return self._run_paratope_task(model_name, model, tokenizer, dataset, label_list, model_config, config, args)
        elif task_name == "cdr_prediction":
            return self._run_cdr_task(model_name, model, tokenizer, dataset, model_config, config, args)
        elif task_name == "her2_prediction":
            return self._run_her2_task(model_name, model, tokenizer, dataset, model_config, config, args)
        elif task_name == "covid_prediction":
            return self._run_covid_task(model_name, model, tokenizer, dataset, model_config, config, args)
        elif task_name == "vh_prediction" or task_name == "vl_prediction":
            return self._run_vh_task(model_name, model, tokenizer, dataset, model_config, config, args)
        else:
            raise ValueError(f"Unknown Task: {task_name}")
    
    def _run_paratope_task(self, model_name, model, tokenizer, dataset, label_list, model_config, config, args):
        batch_size = args.batch_size if args.batch_size else model_config['batch_size']
        lr = args.lr if args.lr else model_config['learning_rate']
        
        training_args = TrainingArguments(
            f"paratope-prediction-task-{model_name}_{args.seed}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=float(lr),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=config['trainer']['num_train_epochs'],
            warmup_ratio=config['trainer']['warmup_ratio'],
            load_best_model_at_end=True,
            lr_scheduler_type=config['trainer']['lr_scheduler_type'],
            metric_for_best_model=config['trainer']['metric_for_best_model'],
            logging_strategy=config['trainer']['logging_strategy'],
            seed=args.seed
        )
        
        def compute_metrics(p):
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
        
        trainer_tokenzier = None if model_name in ['Ablang', 'Ablang2'] else tokenizer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=trainer_tokenzier,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['trainer']['early_stopping_patience']
            )]
        )
        
        trainer.train()
        results = trainer.predict(dataset['test'])
        
        return results.metrics
    
    def _run_cdr_task(self, model_name, model, tokenizer, dataset, model_config, config, args):

        batch_size = args.batch_size if args.batch_size else model_config['batch_size']
        lr = args.lr if args.lr else model_config['learning_rate']
        
        base_args = {
            "output_dir": f"cdr-prediction-task-{model_name}_{args.seed}",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "learning_rate": float(lr),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": 1200,
            "warmup_ratio": config['trainer']['warmup_ratio'],
            "load_best_model_at_end": True,
            "lr_scheduler_type": config['trainer']['lr_scheduler_type'],
            "metric_for_best_model": 'AAR',
            "logging_strategy": config['trainer']['logging_strategy'],
            "seed": args.seed
        }

        # Ablang and Ablang2 need special save method
        if model_name in ['Ablang', 'Ablang2']:
            base_args['save_safetensors'] = False

        training_args = TrainingArguments(**base_args)
        
        def compute_metrics(p):
            predictions, labels = p   
            labels = torch.tensor(labels)
            outputs = torch.tensor(predictions[0])
            infill_mask = predictions[1]

            pred_ids = torch.argmax(outputs, dim=-1)
            num_elem = infill_mask.sum().item()

            matches = labels[infill_mask] == pred_ids[infill_mask]
            succ_matches = matches.sum().item()
            total_matches = len(matches)

            ignore_index = 21 if model_name in ['Ablang', 'Ablang2'] else tokenizer.pad_token_id
            loss_fnc = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            L = loss_fnc(outputs[infill_mask],labels[infill_mask])
            loss_elem = L * num_elem
            accuracy = 100 * (succ_matches / total_matches)

            
            return {
                "AAR": accuracy,
                "loss_elem": loss_elem
            }
        
        trainer_tokenzier = None if model_name in ['Ablang', 'Ablang2'] else tokenizer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=trainer_tokenzier,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['trainer']['early_stopping_patience']
            )]
        )
        
        trainer.train()
        results = trainer.predict(dataset['test'])
        
        return results.metrics
    
    def _run_her2_task(self, model_name, model, tokenizer, dataset, model_config, config, args):
        batch_size = args.batch_size if args.batch_size else model_config['batch_size']
        lr = args.lr if args.lr else model_config['learning_rate']
        
        base_args = {
            "output_dir": f"her2-prediction-task-{model_name}_{args.seed}",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "learning_rate": float(lr),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": 500,
            "warmup_ratio": config['trainer']['warmup_ratio'],
            "load_best_model_at_end": True,
            "lr_scheduler_type": config['trainer']['lr_scheduler_type'],
            "metric_for_best_model": 'acc',
            "logging_strategy": config['trainer']['logging_strategy'],
            "seed": args.seed
        }

        if model_name in ['Ablang', 'Ablang2']:
            base_args['save_safetensors'] = False

        training_args = TrainingArguments(**base_args)
        
        def compute_metrics(p):

            predictions, labels = p  
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
        
        trainer_tokenzier = None if model_name in ['Ablang', 'Ablang2'] else tokenizer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=trainer_tokenzier,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['trainer']['early_stopping_patience']
            )]
        )
        
        trainer.train()
        results = trainer.predict(dataset['test'])
        
        return results.metrics
    
    def _run_covid_task(self, model_name, model, tokenizer, dataset, model_config, config, args):
        batch_size = args.batch_size if args.batch_size else model_config['batch_size']
        lr = args.lr if args.lr else model_config['learning_rate']
        
        base_args = {
            "output_dir": f"covid-prediction-task-{model_name}_{args.seed}",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "learning_rate": float(lr),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": 800,
            "warmup_ratio": config['trainer']['warmup_ratio'],
            "load_best_model_at_end": True,
            "lr_scheduler_type": config['trainer']['lr_scheduler_type'],
            "metric_for_best_model": 'pr_auc',
            "logging_strategy": config['trainer']['logging_strategy'],
            "seed": args.seed
        }

        if model_name in ['Ablang', 'Ablang2']:
            base_args['save_safetensors'] = False

        training_args = TrainingArguments(**base_args)
        
        def compute_metrics(p):

            predictions, labels = p 

            labels = torch.tensor(labels)  
            y_pred = torch.tensor(predictions)

            prob = torch.sigmoid(y_pred)
            y_pred = (prob>0.5).int()

            f1 = metrics.f1_score(y_pred, labels,average='micro')
            precision = metrics.precision_score(y_pred, labels,average='micro')
            recall = metrics.recall_score(y_pred, labels,average='micro')
            total_acc = metrics.accuracy_score(y_pred.view(-1), labels.view(-1))

            pr_auc= metrics.average_precision_score(labels, prob, average='micro')

            return {
                "acc": total_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "pr_auc": pr_auc
            }
        
        trainer_tokenzier = None if model_name in ['Ablang', 'Ablang2'] else tokenizer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=trainer_tokenzier,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['trainer']['early_stopping_patience']
            )]
        )

        trainer.train()
        results = trainer.predict(dataset['test'])
        
        return results.metrics
    
    def _run_vh_task(self, model_name, model, tokenizer, dataset, model_config, config, args):

        batch_size = args.batch_size if args.batch_size else model_config['batch_size']
        # Regression Task need a higher learning rate
        lr = 1e-5
        
        base_args = {
            "output_dir": f"regression-task-{model_name}_{args.seed}",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "learning_rate": float(lr),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": 800,
            "warmup_ratio": config['trainer']['warmup_ratio'],
            "load_best_model_at_end": True,
            "lr_scheduler_type": config['trainer']['lr_scheduler_type'],
            "metric_for_best_model": 'RMSE',
            "logging_strategy": config['trainer']['logging_strategy'],
            "seed": args.seed
        }


        if model_name in ['Ablang', 'Ablang2']:
            base_args['save_safetensors'] = False

        training_args = TrainingArguments(**base_args)
        
        def compute_metrics(p):
            predictions, labels = p   
            labels = torch.tensor(labels)
            outputs = torch.tensor(predictions)
            outputs = torch.squeeze(outputs)
            rmse = np.sqrt(metrics.mean_squared_error(outputs, labels))
            r2 = metrics.r2_score(labels, outputs)  
            mae = metrics.mean_absolute_error(outputs, labels)
            rp = pearsonr(labels, outputs)  


            return {
                "RMSE": rmse,
                "r2": r2,
                "MAE": mae,
                "rp_s": rp[0],
                "rp_p": rp[1]
            }

        trainer_tokenzier = None if model_name in ['Ablang', 'Ablang2'] else tokenizer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=trainer_tokenzier,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config['trainer']['early_stopping_patience']
            )]
        )
        
        trainer.train()
        results = trainer.predict(dataset['test'])
        
        return results.metrics