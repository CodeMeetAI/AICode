import json
import argparse

import numpy as np

def eval(args):
    inference_result = []
    with open(args.result_path, "r") as f:
        for line in f:
            inference_result.append(json.loads(line))
    
    responds = []
    for res in inference_result:
        responds.append(res['answer'][1])

    accuracy = np.mean(np.array(responds) == "A")
    print(accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="../results/frames/llama2_3_turns.jsonl")

    args = parser.parse_args()
    eval(args)