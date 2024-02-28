import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

ROLE_MAPPING = {
    "wizard": "assistant",
    "user": "user"
}

class FramesDatasets:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(data_folder_path, "frames_origin.json")
        self.raw_data = self.load_raw_data()
    
    def load_raw_data(self):
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)
        
        filtered_data = []
        
        for dialogue in raw_data:
            conversations = []
            turns = dialogue['turns']
            label = turns[0]['text']
            for turn in turns:
                json_sample = {
                    "role": ROLE_MAPPING[turn['author']],
                    "content":turn['text'],    
                }
                conversations.append(json_sample)

            filtered_json = {
                "user_id": dialogue['user_id'],
                "conversations": conversations,
                "label": label
            }
            filtered_data.append(filtered_json)        
        
        return filtered_data
            
            
    def save_as_json(self, data, filename="frames_processed.json"):
        with open(os.path.join(self.data_folder_path, filename), 'w') as f:
            json.dump(data, f, indent=4)
        
            
if __name__ == "__main__":
    path = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/frames/"
    dataset = FramesDatasets(data_folder_path=path)
    dataset.save_as_json(data = dataset.raw_data, filename="frames.json")