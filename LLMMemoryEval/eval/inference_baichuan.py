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
            # option_input = torch.tensor(tokenizer.encode(c), device=args.device).unsqueeze(0)
            context.append({"role": "assistant", "content": c})
            option_input = build_chat_input(model, tokenizer, context)
            # option_input = torch.cat([context_input, option_input], dim=1)
            # print(option_input.shape)
            # option_input.insert(0, 1)
            option_input = {"input_ids": option_input}
            with torch.inference_mode(mode=True):
                option_outputs = model(**option_input)
            logits = option_outputs.logits
            log_probs = F.log_softmax(logits[:, 1:, :], dim=-1)
            
            # target_id = option_input['input_ids'][0, -1]
            # log_prob = log_probs[:, target_id].item()
            # option_scores.append(log_prob)
            input_ids = option_input['input_ids']
            # print(input_ids, log_probs.shape)
            gathered_log_probs = torch.gather(log_probs, 2, input_ids[:, log_probs.shape[-2] - 1:].unsqueeze(-1)).squeeze(-1)
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
