import json
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_samples(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def generate_responses_and_save_to_json(tokenizer, model, grouped_samples, output_json_path, 
                                        batch_size=1, show_dialogue=True, mode=1, max_len=128):
    """
    Func:
        Generate the response with llama series model
        
    Args:
        tokenizer: the tokenizer for the model
        model: the llama model object
        grouped_samples (list): a list of input samples. For example, 
                for "mode = 1":
                                    [
                                        [sample 1], [sample 2], 
                                        [
                                            [The first prompt], 
                                            [The second prompt], 
                                            [[prompt], [service label (i.e., "restaurant")]],  ..., 
                                            [The n-th prompt]
                                        ], ...,
                                        [sample n]
                                    ]
                                    
                for "mode = 2":
                                    [
                                        "sample 1", "sample 2", 
                                        "<Your prompt for the third sample>", ...,
                                        "sample n"
                                    ]
        output_json_path: The the path to save output
        show_dialogue: Whether show the output dialogue. Defaults to True.
        mode: 1. chat multiple turns; 2. chat in one turn; 3. generate the text. Defaults to 1.
        max_len: max length for output. Defaults to 128.
    """
    dialogues_output = []
    device = model.device
    for group in grouped_samples:
        group_dialogues_output = []
        for sample in group:
            dialogue, services = sample
            history = None
            dialogue_responses = []
            if mode == 1:
                for turn in dialogue:
                    try:
                        inputs = tokenizer(turn, return_tensors="pt").to(device)
                        output_sequences = model.chat_completion(**inputs, max_length=max_len)
                    except AttributeError:
                        print("An error occurred during generation. Please ensure you have correctly configured the model and tokenizer.")
                        break
            elif mode == 2:
                inputs = tokenizer(sample, return_tensors="pt").to(device)
                output_sequences = model.chat_completion(**inputs, max_length=max_len)
            else:
                inputs = tokenizer(sample, return_tensors="pt").to(device)
                output_sequences = model.text_completion(**inputs, max_length=max_len)
            response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            dialogue_responses.append({"input": turn, "output": response})
            if show_dialogue:
                print("\ninput: "+turn)
                print("output: "+response)
            group_dialogues_output.append(dialogue_responses)
        dialogues_output.append(group_dialogues_output)
        if show_dialogue:
            print(f"Completed group: {len(dialogues_output)}/{len(grouped_samples)}")
        else:
            cnt = 50 * len(dialogues_output) / len(grouped_samples)
            per = "%.2f" % cnt
            done = ">>" * int(cnt // 2)
            remain = "##" * (50 - int(cnt // 2))
            print(f"Chatting: [{done}{remain}] {per}%", end="\r")

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dialogues_output, json_file, ensure_ascii=False, indent=4)
    print("\nTask is done!")

# Assuming the token and model paths are correct and valid
token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", trust_remote_code=True, token=token).eval()

samples_file_path = './grouped_samples.pkl'
grouped_samples = load_samples(samples_file_path)

output_json_path = './grouped_dialogues_output.json'

generate_responses_and_save_to_json(tokenizer, model, grouped_samples, output_json_path, show_dialogue=False, chat=True, max_len=128)
