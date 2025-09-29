# utils.py - 工具函数模块
import torch
import numpy as np
import random
import os
import json
from datetime import datetime

def set_seed(seed: int = 42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_results(results, model_name, task_name):
    """保存实验结果"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{model_name}_{task_name}_{timestamp}.json"
    
    result_data = {
        "model": model_name,
        "task": task_name,
        "timestamp": timestamp,
        "metrics": results
    }
    
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=4)
    
    print(f"结果已保存到: {filename}")