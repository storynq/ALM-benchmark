# ALM-benchmark
We present a comprehensive benchmark to evaluate the capabilities of eight **Antibody Language Models (ALMs)** acorss multiple antibody design tasks.

# Datasets
All datasets used in this study are provided in `./dataset`. Downstream task datasets are clustered using mmseqs2 and then divided into training, validation and testing sets based on stratified sampling.
* Representation_learning: Data about six IGHV subtypes for ALMs' representation learning
* Paratope: Data for paratope prediction
* CDR: Data for CDR task
* Her2: Data for HER2 binding prediction
* Covid: Datasets for Covid binding prediction
* VH & VL: Datasets for binding affinity prediction
* Bert2DAb requires all datasets with tokens based on secondary structure, which are provided in `Bert2Dab_Dataset`.

# Environment&Pretrained_files
The required dependenices can be installed via `pip` or `conda` from `requirements.txt`

Code environment and some pretrained model files are provided on Zenodo, which can be download from this links: `ZZZZZZZENODO LINK!`

# Codes
Fine-tuning codes for ALMs on five downstream tasks are provided in `./codes`.

## Basic usage
```
main.py --model <model_name> --task <task_name> [options]
```
## Required Arguments
* ```--model```: Name of the model to evaluate
Examples: antiberta, antiberta2, Bert2DAb, etc.
* ```--task```: Name of the task to execute
Examples: paratope_prediction, cdr_prediction, etc.

## Optional Arguments
* ```--config```: Path to configuration file
* ```--seed```: Random seed
* ```--lr```: Learning rate (default: 1e-6), for vh and vl prediction task, 1e-5 was used.
* ```--batch_size```: Batch size
* ```--cdr_type```: CDR type for cdr_prediction task. Default: CDR1. Other choices: CDR2, CDR3

## Usage Exapmles:
### For paratope prediction task
```
python main.py --model antiberta --task paratope_prediction
```
### For CDR prediction task
```
python main.py --model antiberta2 --task cdr_prediction --cdr_type CDR3
```

Make sure all dependencies are installed and the model paths/data paths are correctly configured in ```config.yaml``` before running the framework


