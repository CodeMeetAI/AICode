import json
import argparse
import os
import csv
import pandas as pd

def calculate_ap(gt, options_dist):
    sorted_options = sorted(options_dist, key=lambda x: x[1], reverse=True)
    correct_index = [i for i, option in enumerate(sorted_options) if option[0] == gt][0]
    return 1 / (correct_index + 1)

def eval(result_path):
    total_count = 0
    correct_count = 0
    ap_list = []
    option_accuracy = {'A': [0, 0], 'B': [0, 0], 'C': [0, 0]} 
    options_choice = {'A': 0, 'B': 0, 'C': 0} 
    category_counts = {'hotel': [0, 0], 'train': [0, 0], 'attraction': [0, 0], 'restaurant': [0, 0]}  # [correct, total]

    # detect whether result_path is a directory
    if os.path.isdir(result_path):
        return None
    
    with open(result_path, 'r', encoding='utf-8') as file:
        cnt = 0
        for line in file:
            data = json.loads(line)

            total_count += 1
            correct_option = data['gt'].split(')')[0][1]  # A, B, C
            answer_option = data['answer'].split(')')[0][1] 
            # correct_option = data['gt'].split(".")[0]  # A, B, C
            # answer_option = data['answer'].split(".")[0] 
            gt_category = data['gt'].split('.')[1]
            option_accuracy[correct_option][1] += 1
            
            if data['answer'] == data['gt']:
                correct_count += 1
                option_accuracy[correct_option][0] += 1 
                category_counts[gt_category][0] += 1
            if (ans := data['answer'].split(".")[1]) != "hospital":
                category_counts[ans][1] += 1
            else:
                category_counts[list(category_counts.keys())[cnt%4]][1] += 1
                cnt += 1 
            
            options_choice[answer_option] += 1
            ap = calculate_ap(data['gt'], data['options_dist'])
            ap_list.append(ap)


    accuracy = round(correct_count / total_count, 3) if total_count > 0 else 0
    map_score = round(sum(ap_list) / len(ap_list),3) if ap_list else 0
    option_accuracy_rates = {opt: (round(correct / total,3) if total > 0 else 0) for opt, (correct, total) in option_accuracy.items()}
    category_accuracies = {category: round(correct / total,3) if total > 0 else 0 for category, (correct, total) in category_counts.items()}
    options_choice_prop = {opt: (round(opt_count / total_count,3))  for opt, opt_count in options_choice.items()}
    category_prop = {category: (round(total / total_count,3))  for category, (_, total) in category_counts.items()}
    print(category_prop)

    return accuracy, map_score,option_accuracy_rates,category_accuracies, options_choice_prop, category_prop

def save_results_to_csv(results, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'turn', 'key', 'acc_score', 'map_score', 'acc_A', 'acc_B', 'acc_C', 
                      'hotel_acc', 'train_acc', 'attraction_acc', 'restaurant_acc', "A", "B", "C", "hotel", 
                      "train", "attraction", "restaurant"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'model_name': result[0], 'turn': result[1], 'key': result[2], 'acc_score': result[3], 'map_score': result[4],
                            'acc_A': result[5]['A'], 'acc_B': result[5]['B'], 'acc_C': result[5]['C'],
                            'hotel_acc': result[6]['hotel'], 'train_acc': result[6]['train'], 'attraction_acc': result[6]['attraction'], 'restaurant_acc': result[6]['restaurant'],
                            "A": result[7]['A'], "B": result[7]['B'], 'C': result[7]["C"],
                            "hotel": result[8]['hotel'], "train": result[8]['train'], 'attraction': result[8]["attraction"], 'restaurant': result[8]["restaurant"]})
    
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
            acc_score,map_score,option_accuracy_rates,category_accuracies, options_choice_prop, category_prop = eval(full_inference_file)
            if acc_score is None:
                continue    
            if map_score is None:
                continue
            parts = inference_file.split("-")
            model_name = parts[0]
            turn = parts[1]
            key = "-".join(parts[:3])

            results.append((model_name, turn, key, acc_score, map_score,option_accuracy_rates,category_accuracies, options_choice_prop, category_prop))
    
    save_results_to_csv(results, output_file)