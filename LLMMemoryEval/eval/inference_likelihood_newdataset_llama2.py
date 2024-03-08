import json
import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

def inference(args):
    token = "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"
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=token)

    options = json.load(open(args.option_file, 'r'))
    contexts = json.load(open(args.context_file, 'r'))
    
    answers_file = args.answers_file + datetime.now().strftime("-%Y-%m-%d-%H-%M") + ".jsonl"
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for option, context in tqdm(zip(options, contexts)):
        prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        context_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
        with torch.inference_mode(mode=True):
            context_outputs = model(**context_input, use_cache=True)
        past_key_values = context_outputs.past_key_values
        option_scores = []
        for c in option['options']:
            option_input = tokenizer(c, return_tensors='pt', add_special_tokens=True).to(args.device)
            with torch.inference_mode(mode=True):
                option_outputs = model(**option_input, past_key_values=past_key_values)
            logits = option_outputs.logits
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            target_id = option_input['input_ids'][0, -1]
            log_prob = log_probs[:, target_id].item()
            option_scores.append(log_prob)
        
        best_option_index = option_scores.index(max(option_scores))
        best_option = option['options'][best_option_index]
        gt = option['gt']

        ans_file.write(json.dumps({"answer": best_option, "gt": gt}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_file", type=str)
    parser.add_argument("--option_file", type=str)
    parser.add_argument("--answers_file", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    inference(args)
