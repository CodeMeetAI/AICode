import json
import argparse
import re
import numpy as np

def eval(args, correct_answer="A"):
    inference_result = []
    answer_choice = "ABCDEF"
    try:
        if "first" in args.result_path:
            correct_answer = "A"
        elif "mid" in args.result_path:
            num = int(args.result_path.split("/")[-1].split("_")[1])
            correct_answer = answer_choice[num//2]
        else:
            num = int(args.result_path.split("/")[-1].split("_")[1])
            correct_answer = answer_choice[num - 1]
        with open(args.result_path, "r") as f:
            for line in f:
                inference_result.append(json.loads(line))
        
        responds = []
        for res in inference_result:
            answer = re.search(r"[ABCDEF]", res['answer'])
            if answer:  
                responds.append(answer.group(0))
            else:
                responds.append("X")
        assert len(responds) == len(inference_result), "The length is incorrect."
        accuracy = np.mean(np.array(responds) == correct_answer)
        print(accuracy)
    except:
        print("No data yet")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="../results/frames/llama2_3_turns.jsonl")

    args = parser.parse_args()
    eval(args)