import json
import argparse
import os
from collections import defaultdict
import csv

def eval(result_path):
    total_count = 0
    correct_count = 0

    # detect whether result_path is a directory
    if os.path.isdir(result_path):
        return None
    
    with open(result_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            total_count += 1
            if data['answer'] == data['gt']:
                correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    return accuracy

def save_results_to_csv(results, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'turn', 'key', 'metric_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'model_name': result[0], 'turn': result[1], 'key': result[2], 'metric_score': result[3]})
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_root", type=str, required=True)
    args = parser.parse_args()
    
    results_file_root = args.result_file_root
    output_file = results_file_root+"/results.csv"
    
    results = []
            
    for inference_file in os.listdir(results_file_root):
        full_inference_file = os.path.join(results_file_root, inference_file)
        if full_inference_file.endswith(".jsonl"):
            metric_score = eval(full_inference_file)
            if metric_score is None:
                continue    

            parts = inference_file.split("-")
            model_name = parts[0]
            turn = parts[1]
            key = "-".join(parts[:3])

            results.append((model_name, turn, key, metric_score))
    
    save_results_to_csv(results, output_file)