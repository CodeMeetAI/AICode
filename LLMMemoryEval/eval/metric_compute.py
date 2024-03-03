import json
import argparse
import re
import numpy as np

def eval(args):
    inference_result = []
    correct_answers = []

    with open(args.answer_path, "r") as f:
        correct_answers_file = json.load(f)
        correct_answers = []
        for item in correct_answers_file:
            correct_answer = re.search(r"[ABCDEF]", item['user input'])
            correct_answers.append(correct_answer.group(0))
    
    with open(args.result_path, "r") as f:
        for line in f:
            inference_result.append(json.loads(line))
    
    assert len(inference_result) == len(correct_answers)
    
    responds = []
    for res in inference_result:
        answer = re.search(r"[ABCDEF]", res['answer'])
        if answer:  
            responds.append(answer.group(0))
        else:
            responds.append("X")
    assert len(responds) == len(inference_result), "The length is incorrect."
    accuracy = np.mean(np.array(responds) == np.array(correct_answers))
    #print(responds)
    #print(correct_answers[0:1000])
    print(accuracy)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="../results/frames/llama2_3_turns.jsonl")
    parser.add_argument("--answer_path", type=str, default="../datasets/data/frames/labels/frames_grouped_2_first-label.json")
    args = parser.parse_args()
    eval(args)