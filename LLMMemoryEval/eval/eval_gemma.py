import argparse
from transformers import AutoTokenizer, GemmaForCausalLM
import json
import os



def eval(args):
    token = "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"
    model = GemmaForCausalLM.from_pretrained("google/gemma-7b-it", token=token).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=token)

    print("model loaded")
    
    with open(args.data_dir, "r") as f:
        conversations = json.load(f)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    print("start inference")
    for grouped_conversation in conversations:
        prompt = tokenizer.apply_chat_template(grouped_conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = 8)
        out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("answer: ", out)
        ans_file.write(json.dumps({"answer": out}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/data/frames/frames_multiple_choice_3.json")
    parser.add_argument("--answers_file", type=str, default="../results/frames/gemma_3_turns.jsonl")

    args = parser.parse_args()
    eval(args)