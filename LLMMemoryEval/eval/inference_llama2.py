import argparse
import json
import os
import torch
from datetime import datetime

from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

def eval(args):
    token = "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"
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token)

    tokenizer.pad_token = tokenizer.eos_token
    print("model loaded")
    
    answers_file = args.answers_file + datetime.now().strftime("-%m-%d-%H-%m") + ".jsonl"
    
    with open(args.data_dir, "r") as f:
        conversations = json.load(f)
    
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    print("start inference")
    for grouped_conversation in tqdm(conversations):
        prompt = " ".join(message["content"] for message in grouped_conversation)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(args.device)
        
        options = None
        option_inputs = tokenizer(options, return_tensor='pt', padding=True)
        
        with torch.no_grad():
            context_output = model(input_ids=inputs.input_ids, max_new_tokens=8)
            # generate_ids = model.generate(inputs.input_ids, max_new_tokens = 8)
            past_key_values = context_output.past_key_values

        likelihood = []
        option_output = model(option_inputs.input_ids, past_key_values=past_key_values)
        
        option_output

        ans = likelihood.index(min(likelihood))

        ans_file.write(json.dumps({"answer": ans}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/data/frames/frames_multiple_choice_3.json")
    parser.add_argument("--answers_file", type=str, default="../results/frames/llama2_3_turns")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    eval(args)
    