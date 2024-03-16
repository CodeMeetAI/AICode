import torch
import json
import os
import argparse
import numpy as np

# import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from datetime import datetime
from typing import List
from tqdm import tqdm
from copy import deepcopy

def inference(args):
    token = args.token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, token=token, trust_remote_code=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=token, trust_remote_code=True)

    options = json.load(open(args.option_file, 'r'))
    contexts = json.load(open(args.context_file, 'r'))
    
    answers_file = args.answers_file + datetime.now().strftime("-%m-%d-%H-%M") + ".jsonl"
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    model.eval()

    for option, context in tqdm(zip(options, contexts)):
        option_scores = []
        for c in option['options']:
            tmp_context = deepcopy(context)
            if tmp_context[-1]["role"] == "user":
                tmp_context[-1]["content"] += c
            else:
                tmp_context = tmp_context + [{"role": "user", "content": c}]
            prompt        = tokenizer.apply_chat_template(tmp_context, tokenize=False, add_generation_prompt=True)
            context_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
            input_ids     = context_input['input_ids']
            prompt_length = context_input.input_ids.size(1)
            option_length = len(tokenizer.encode(c, add_special_tokens=False))
            target_ids    = input_ids[0, -option_length:]
            with torch.inference_mode(mode=True):
                outputs = model.generate(
                    **context_input,
                    max_length=prompt_length + 2, # + len(tokenizer.encode(c, add_special_tokens=False))+8,
                    return_dict_in_generate=True,
                    output_logits=True,
                    # max_new_tokens=1,
            )
                
            # transition_scores = model.compute_transition_scores(
            #     outputs.sequences,
            #     outputs.scores,
            #     normalize_logits=True
            # )
            
            logits = outputs.logits  # Get the logits for the last token position
            # print(tokenizer.decode(outputs.sequences[0]))
            # print(f"logits  {logits[-2:][0]}")
            logits = torch.stack(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # print(log_probs.shape, input_ids.shape, prompt_length)
            gathered_log_probs = torch.gather(log_probs[:, ], 2, input_ids[:, target_ids].unsqueeze(-1)).squeeze(-1)
            # print(f"Prob of '{c}': {gathered_log_probs}")
            average_log_prob = torch.mean(gathered_log_probs)
            option_scores.append(average_log_prob.item())
            
            # for tok, score in zip(generated_tokens[0], transition_scores[0]):
            #     #tokentoken stringlog probabilityprobability
            #     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")

        
    #     best_option_index = option_scores.index(max(option_scores))
    #     print(f"option_scores {option_scores}")
    #     best_option = option['options'][best_option_index]
    #     gt = option['gt']

    #     result = {
    #         "answer": best_option,
    #         "gt": gt,
    #         "options_dist": list(zip(option['options'], option_scores))
    #     }
    #     #print(result)
    #     ans_file.write(json.dumps(result) + "\n")
    #     ans_file.flush()

    # ans_file.close()        

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
