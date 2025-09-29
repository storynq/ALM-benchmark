import argparse
import yaml
from data_processing import DataProcessor
from model_loading import ModelLoader
from task_runner import TaskRunner
from utils import set_seed, save_results
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Antibody Language Models Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Model Name, e.g. antiberta')
    parser.add_argument('--task', type=str, required=True, help='Task Name, e.g. paratope_prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='config path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--cdr_type', type=str, choices=['CDR1', 'CDR2', 'CDR3'])
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    set_seed(args.seed)
    
    model_config = config['models'][args.model]
    if args.lr:
        model_config['learning_rate'] = args.lr
    if args.batch_size:
        model_config['batch_size'] = args.batch_size

    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model(args.model, args.task, model_config)
    

    data_processor = DataProcessor()
    if args.task.startswith("cdr"):
        if not args.cdr_type:
            cdr_type = "CDR1"
            print(f"cdr_type not found, CDR1 used: {cdr_type}")
        else:
            cdr_type = args.cdr_type
            
        dataset, label_list = data_processor.process_data(
            args.model, args.task, config, tokenizer, cdr_type
        )
    else:
        dataset, label_list = data_processor.process_data(
            args.model, args.task, config, tokenizer
        )
    
    print(model)
    task_runner = TaskRunner()
    results = task_runner.run_task(
        args.model, args.task, model, tokenizer, dataset, label_list, model_config, config, args
    )
    
    save_results(results, args.model, args.task)
    
    print(f"Task_Finished!: {args.model} in {args.task} ")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()