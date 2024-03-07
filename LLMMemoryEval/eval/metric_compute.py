import json
import argparse
import os
from collections import defaultdict

import pandas as pd

def eval(result_path):
    total_count = 0
    correct_count = 0

    with open(result_path, 'r', encoding='utf-8') as file:
        print(result_path)
        for line in file:
            data = json.loads(line)

            total_count += 1
            if data['answer'] == data['gt']:
                correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(accuracy)
    return accuracy
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_root", type=str, default="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results/dialogue_based")
    args = parser.parse_args()
    
    resuls_file_root = args.result_file_root
    
    results = defaultdict(list)
    
    for position_dir in os.listdir(resuls_file_root):
        if "csv" in position_dir:
            continue
        full_position_dir = os.path.join(resuls_file_root, position_dir)
        
        for dataset_dir in os.listdir(full_position_dir):
            full_dataset_dir = os.path.join(full_position_dir, dataset_dir)
            
            for inference_file in os.listdir(full_dataset_dir):
                if "csv" in inference_file:
                    continue
                full_inference_file = os.path.join(full_dataset_dir, inference_file)
                metric_score = eval(full_inference_file)

                model_name = inference_file.split("_")[0]
                turn = inference_file.split("_")[1]
                
                results[model_name].append({
                    'position': position_dir,
                    'dataset': dataset_dir,
                    'turn': turn,
                    'metric scores': metric_score
                })
    
    for model_name, res in results.items():
        df = pd.DataFrame(res)
        df.to_csv(resuls_file_root + model_name + ".csv")
    
    # eval(args.result_path)