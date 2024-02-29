import argparse
import json
import os
from datetime import datetime

from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm

def eval(args):
    token = "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"
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token = token)

    tokenizer.pad_token = tokenizer.eos_token
    print("model loaded")
    
    answers_file = args.answers_file + datetime.now().strftime("-%m-%d-%H-%m") + f"_{args.data_dir.split("_")[-1].split(".")[0]}.jsonl"
    
    with open(args.data_dir, "r") as f:
        conversations = json.load(f)
    
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    print("start inference")
    for grouped_conversation in tqdm(conversations):
        prompt = " ".join(message["content"] for message in grouped_conversation)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(args.device)
        generate_ids = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1] + 8)
        out = tokenizer.decode(generate_ids[:, inputs.input_ids.shape[1]:][0], skip_special_tokens=True)
        ans_file.write(json.dumps({"answer": out}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/data/frames/frames_multiple_choice_3.json")
    parser.add_argument("--answers_file", type=str, default="../results/frames/llama2_3_turns")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    eval(args)
    