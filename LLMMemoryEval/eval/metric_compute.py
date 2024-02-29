import json
import argparse
import re
import numpy as np

def eval(args, correct_answer="A"):
    inference_result = []
    with open(args.result_path, "r") as f:
        for line in f:
            inference_result.append(json.loads(line))
    
    responds = []
    for res in inference_result:
        answer = re.search(r"\([^A-Z]*([A-Z])[^A-Z]*\)",res['answer'])
        if answer:  
            responds.append(answer.group(1))

    accuracy = np.mean(np.array(responds) == correct_answer)
    print(accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="../results/frames/llama2_3_turns.jsonl")

    args = parser.parse_args()
    eval(args)