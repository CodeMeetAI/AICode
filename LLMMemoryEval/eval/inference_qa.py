import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch.nn.functional as F
import json
from datetime import datetime
import os
import argparse
from tqdm import tqdm

def inference(args):
    token = args.token
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path, token=token, trust_remote_code=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=token, trust_remote_code=True)
    

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
            log_probs = F.log_softmax(logits[:, 1:, :], dim=-1)
            
            # target_id = option_input['input_ids'][0, -1]
            # log_prob = log_probs[:, target_id].item()
            # option_scores.append(log_prob)
            input_ids = option_input['input_ids']
            gathered_log_probs = torch.gather(log_probs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            average_log_prob = torch.mean(gathered_log_probs)
            option_scores.append(average_log_prob.item())
        
        best_option_index = option_scores.index(max(option_scores))
        best_option = option['options'][best_option_index]
        gt = option['gt']

        ans_file.write(json.dumps(
            {
                "answer": best_option,
                "gt": gt,
                "options_dist": list(zip(option['options'], option_scores))
             }
        ) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_file", type=str, required=True)
    parser.add_argument("--option_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--token", type=str, default="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")
    
    args = parser.parse_args()
    inference(args)
