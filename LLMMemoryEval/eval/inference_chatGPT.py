import json
import os
import requests
from datetime import datetime
from tqdm import tqdm
import argparse


API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "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"

def call_openai_api(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    json_data = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": 50, 
        "n": 1,
        "logprobs": 1, 
    }

    response = requests.post(API_URL, headers=headers, json=json_data)
    if response.status_code != 200:
        print(f"API request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None

    response_data = response.json()
    if 'error' in response_data:
        print("Error in API response:", response_data['error'])
        return None

    return response_data

def calculate_logprobs_for_options(contexts, options):
    context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in contexts])
    option_scores = []
    for option in options:
        prompt = context_text + f"\nAI: {option}"
        api_response = call_openai_api(prompt)

        logprobs = api_response['choices'][0]['logprobs']['token_logprobs']
        total_logprob = sum(logprobs)
        option_scores.append(total_logprob)
    return option_scores

def inference(args):
    options_list = json.load(open(args.option_file, 'r'))
    contexts_list = json.load(open(args.context_file, 'r'))
    
    answers_file = args.answers_file + datetime.now().strftime("-%Y-%m-%d-%H-%M") + ".jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for options, contexts in tqdm(zip(options_list, contexts_list), total=len(options_list)):
            option_scores = calculate_logprobs_for_options(contexts, options['options'])

            best_option_index = option_scores.index(max(option_scores))
            best_option = options['options'][best_option_index]

            ans_file.write(json.dumps({
                "context": contexts,
                "options": options['options'],
                "selected_option": best_option,
                "option_scores": option_scores,
                "gt": options['gt']
            }) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_file", type=str, required=True)
    parser.add_argument("--option_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    
    args = parser.parse_args()
    inference(args)