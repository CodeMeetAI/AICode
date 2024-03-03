import argparse
from datetime import datetime
import json
import os
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, GemmaForCausalLM

def eval(args):
    token = "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"
    model = GemmaForCausalLM.from_pretrained("google/gemma-7b-it", token=token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=token)

    print("model loaded")
    
    answers_file = args.answers_file + datetime.now().strftime("-%m-%d-%H-%m") + ".jsonl"
    
    with open(args.data_dir, "r") as f:
        conversations = json.load(f)
    
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    print("start inference")
    for grouped_conversation in tqdm(conversations):
        prompt = tokenizer.apply_chat_template(grouped_conversation, tokenize=False, add_generation_prompt=True)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids, max_new_tokens = 8)
        # out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(out)
        # filtered_out = out.split("\nAnswer:\nmodel\n")[-1].split("\n")[0].split(".")[0]
        out = tokenizer.decode(generate_ids[:, inputs.input_ids.shape[1]:][0], skip_special_tokens=True)
        
        ans_file.write(json.dumps({"answer": out}) + "\n")
        ans_file.flush()
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         print("Skipping data due to CUDA OOM error.")
        #         # empty the CUDA memory
        #         torch.cuda.empty_cache()
        #         continue
        #     else:
        #         raise e 
    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/data/frames/frames_multiple_choice_3.json")
    parser.add_argument("--answers_file", type=str, default="../results/frames/gemma_3_turns")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    eval(args)