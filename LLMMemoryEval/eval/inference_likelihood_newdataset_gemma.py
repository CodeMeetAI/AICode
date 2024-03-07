from transformers import AutoTokenizer, GemmaForCausalLM
import torch
import torch.nn.functional as F
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm

def inference(args):
    token = "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"
    model = GemmaForCausalLM.from_pretrained("google/gemma-7b-it", token=token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=token)

    options = json.load(open(args.option_file, 'r'))
    contexts = json.load(open(args.context_file, 'r'))
    
    answers_file = args.answers_file + datetime.now().strftime("-%Y-%m-%d-%H-%M") + ".jsonl"
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for option, context in tqdm(zip(options, contexts)):
        prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        context_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
        option_scores = []
        for c in option['options']:
            combined_input = tokenizer(prompt + c, return_tensors='pt', add_special_tokens=True).to(args.device)
            with torch.no_grad():
                outputs = model(**combined_input)
            logits = outputs.logits
            # 直接计算最后一个token的log softmax，代表该选项的概率
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            # 获取与输入对应的log_probs
            target_id = combined_input['input_ids'][0, -1]
            log_prob = log_probs[:, target_id].item()
            option_scores.append(log_prob)
            print(option_scores)
        
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
