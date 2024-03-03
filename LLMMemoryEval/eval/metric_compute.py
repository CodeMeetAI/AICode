import json
import argparse
import re
import numpy as np

def eval(args):
    total_count = 0
    correct_count = 0

    with open(args.result_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            total_count += 1
            if data['answer'] == data['gt']:
                correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f'acc: {accuracy}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results/likelihood_exp/first/frames/gemma_4_new_likelihood-03-03-15-03.jsonl")
    args = parser.parse_args()
    eval(args)