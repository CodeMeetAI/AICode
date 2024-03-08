import json
import os
import numpy as np
import pandas as pd


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
class DatasetBuilder:
    """
    To build our own QA reasoning dataset from the original multiwoz dataset.
    
    Args:
        dir_pth (str): the directory path of the original multiwoz dataset.
    """
    def __init__(self, dir_pth = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/multiwoz/original data",):
        self.data = []
        _ = self.load_from_dir(dir_pth)
        
    def load_from_dir(self, dir_pth, propotion = 1):
        """
        Load all the json files from a directory and return a list of json data.
        
        Args:
            propotion (float): the propotion of the data to be loaded. Default is 1.
        """
        for root, _, files in os.walk(dir_pth):
            for file in files:
                if file.endswith(".json"):
                    self.data = self.data + load_json(os.path.join(root, file))
        return self.data[:int(len(self.data) * propotion)]
    
    def find_active(self, turn):
        """
        Return the active intent of a conversation
        """
        for t in turn:
            try:
                if t["state"]["active_intent"] != "NONE":
                    return t["state"]["active_intent"]
            except:
                return None
        return None
    
    def build_qa_reasoning_dataset(self, data, save_pth):
        """
        Build our own QA reasoning dataset from the original multiwoz dataset.
        
        Args:
            data (list): the original multiwoz dataset.
            save_pth (str): the directory path to save the new dataset.
        """
        new_data = []
        pass # TODO: add the reasoning process here
        save_json(save_pth, new_data)