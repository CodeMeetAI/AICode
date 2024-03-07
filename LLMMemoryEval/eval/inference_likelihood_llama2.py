import json
import os
import argparse
from datetime import datetime

import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm


def inference(args):
    token = "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"
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token)

    options = json.load(open(args.option_file, 'r'))
    choices = list(map(lambda x: x['options'], options))
    labels = list(map(lambda x: x['gt'], options))

    contexts = json.load(open(args.context_file, 'r'))
    
    answers_file = args.answers_file + datetime.now().strftime("-%m-%d-%H-%m") + ".jsonl"
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for choice, context, label in tqdm(zip(choices, contexts, labels)):
        prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        
        context_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(**context_input, use_cache=True)
        past_key_values = outputs.past_key_values
        option_scores = []
        for c in choice:
            choice_input = tokenizer(c, return_tensors='pt', add_special_tokens=False).to(args.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**choice_input, past_key_values=past_key_values)
            
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = choice_input['input_ids']
            shifted_input_ids = input_ids[:, 1:]
            gathered_log_probs = torch.gather(log_probs[:, :-1], 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
            average_log_prob = gathered_log_probs.sum(dim=1) / shifted_input_ids.sum(dim=1)
            option_scores.append(average_log_prob.item())
        best_option_index = option_scores.index(max(option_scores))
        best_option = choice[best_option_index]

        ans_file.write(json.dumps({"answer": best_option, "gt": label}) + "\n")
        ans_file.flush()
    ans_file.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_file", type=str, default="../datasets/data/frames/frames_grouped_new_4_first.json")
    parser.add_argument("--option_file", type=str, default="../datasets/data/frames/labels/frames_grouped_new_4_first-label.json")
    parser.add_argument("--answers_file", type=str, default="../results/first/frames/gemma_4_new_likelihood")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    inference(args)
