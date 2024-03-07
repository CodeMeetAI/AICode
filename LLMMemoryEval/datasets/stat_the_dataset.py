import json
import os
import re
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GemmaForCausalLM
from tqdm import tqdm

datasets_pth = "./datasets/data"
datasets_dir = {}


def get_token_cnt(text):
    notation = r"""['".!?,;:\(\)\[\]{}\<\>]"""
    cnt = len(re.findall(notation, text))
    text = re.sub(notation, " ", text).replace("  ", "")
    return len(text.split(" ")) + cnt

def stat():
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=token)
    
    for dataset in ["frames", "multiwoz", "natural_questions"]:
        datasets_dir[dataset] = {"file": 0, "conversation": 0, "turn": 0}
        stat_num = [0, 0, 0]
        total = 0
        print("**" * 50)
        for file in os.listdir(f"{datasets_pth}/{dataset}"):
            if bool(re.search(r'\d', file)):
                file_total = 0
                file_stat_num = [0, 0]
                max_token = [0, 0]
                
                with open(f"{datasets_pth}/{dataset}/{file}", "r") as f:
                    conversations = json.load(f)
                    stat_num[0] += 1
                    for conversation in conversations:
                        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                        print(prompt)
                        input()
                        # conversation_token = 0
                        stat_num[1] += 1
                        file_stat_num[0] += 1
                        # for turn in conversation:
                        #     stat_num[2] += 1
                        #     file_stat_num[1] += 1
                        #     turn_token = get_token_cnt(turn["content"]tokenizer(prompt, return_tensors="pt", add_special_tokens=False))
                        #     conversation_token += turn_token
                        #     file_total += turn_token
                        #     if turn_token > max_token[1]:
                        #         max_token[1] = turn_token
                        conversation_token = get_token_cnt(prompt)
                        if conversation_token > max_token[0]:
                            max_token[0] = conversation_token
                    total += file_total
                print(f"{file}: {file_total}")
                print(f"\tconversation: {file_total/file_stat_num[0]}")
                print(f"\tmax conversation: {max_token[0]}")
                # print(f"\tmax turn: {max_token[1]}")
                # print(f"\tturn: {file_total/file_stat_num[1]}")
        print("**" * 50)
        datasets_dir[dataset]["file"] = total/stat_num[0]
        datasets_dir[dataset]["conversation"] = total/stat_num[1]
        # datasets_dir[dataset]["turn"] = total/stat_num[2]
        print(f"{dataset}: {datasets_dir[dataset]}")
    
    
if __name__ == '__main__':
    stat()





