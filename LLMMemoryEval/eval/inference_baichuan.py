import torch
import torch.nn.functional as F
import json
import os
import argparse
# import inspect
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from typing import List
from tqdm import tqdm
from copy import deepcopy


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)

def inference(args):
    token = args.token    
    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
        revision="v2.0",
        use_fast=False,
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
        revision="v2.0",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", revision="v2.0")

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
            prompt = tokenizer.apply_chat_template(tmp_context, tokenize=False, add_generation_prompt=True)
            context_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
            with torch.inference_mode(mode=True):
                option_outputs = model(**context_input, labels=context_input["input_ids"])
            average_log_prob = option_outputs.loss * -1
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
