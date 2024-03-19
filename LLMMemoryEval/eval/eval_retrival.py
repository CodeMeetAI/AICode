import json
import argparse
import os
import csv
import pandas as pd
import re

def extract_option_label(full_text):
    match = re.match(r"\((.*?)\)", full_text)
    if match:
        return match.group(1) 
    return None

def calculate_ap(gt, options_dist):
    sorted_options = sorted(options_dist, key=lambda x: x[1], reverse=True)
    correct_index = [i for i, option in enumerate(sorted_options) if option[0] == gt][0]
    return 1 / (correct_index + 1)

def eval(result_path):
    total_count = 0
    correct_count = 0
    ap_list = []
    option_accuracy = {'A': [0, 0], 'B': [0, 0], 'C': [0, 0]} 
    category_counts = {'hotel': [0, 0], 'train': [0, 0], 'attraction': [0, 0], 'restaurant': [0, 0]}  # [correct, total]

    # detect whether result_path is a directory
    if os.path.isdir(result_path):
        return None
    
    with open(result_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            answer_label = extract_option_label(data['answer'])
            gt_label = extract_option_label(data['gt'])

            total_count += 1
            if answer_label and gt_label:
                option_accuracy[gt_label][1] += 1  
                if answer_label == gt_label:
                    correct_count += 1
                    option_accuracy[gt_label][0] += 1  
            
            ap = calculate_ap(data['gt'], data['options_dist'])
            ap_list.append(ap)


    accuracy = round(correct_count / total_count, 3) if total_count > 0 else 0
    map_score = round(sum(ap_list) / len(ap_list),3) if ap_list else 0
    option_accuracy_rates = {opt: (round(correct / total,3) if total > 0 else 0) for opt, (correct, total) in option_accuracy.items()}


    return accuracy, map_score,option_accuracy_rates

def save_results_to_csv(results, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'turn', 'key', 'acc_score', 'map_score', 'acc_A', 'acc_B', 'acc_C']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'model_name': result[0], 'turn': result[1], 'key': result[2], 'acc_score': result[3], 'map_score': result[4],
                            'acc_A': result[5]['A'], 'acc_B': result[5]['B'], 'acc_C': result[5]['C']
                            })
    
        df = pd.DataFrame(list(map(lambda x: dict(zip(fieldnames, x)),  results)))
        print(df)
        df.T.to_excel("results_t.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_root", type=str, required=True)
    args = parser.parse_args()
    
    results_file_root = args.result_file_root
    output_file = results_file_root+"/results.csv"
    
    results = []
            
    for inference_file in sorted(os.listdir(results_file_root)):
        full_inference_file = os.path.join(results_file_root, inference_file)
        if full_inference_file.endswith(".jsonl"):
            acc_score,map_score,option_accuracy_rates = eval(full_inference_file)
            if acc_score is None:
                continue    
            if map_score is None:
                continue
            parts = inference_file.split("-")
            model_name = parts[0]
            turn = parts[1]
            key = "-".join(parts[:3])

            results.append((model_name, turn, key, acc_score, map_score,option_accuracy_rates))
    
    save_results_to_csv(results, output_file)