# ALM-benchmark
We present a comprehensive benchmark to evaluate the capabilities of eight **Antibody Language Models (ALMs)** acorss multiple antibody design tasks.

# Datasets
All datasets used in this study are provided in `./dataset`:
* `/Paratope`: Data for paratope prediction
* CDR_prediction: Data for CDR task
* Her2_dataset: Data for HER2 binding prediction
* SARS-COV1 and SARS-COV2: Datasets for Covid binding prediction
* VH & VL: Datasets for binding affinity prediction

# Environment
The required dependenices can be installed via `pip` or `conda` from `requirements.txt`

# Codes
Pretraining code and fine-tuning codes for Antiberta on five downstream tasks are provided in `./codes`.
