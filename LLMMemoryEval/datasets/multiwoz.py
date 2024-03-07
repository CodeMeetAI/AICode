import json
import os
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

@dataclass
class Turn:
    uid: str
    active_intent: str
    service: str
    slot_values: Dict[str, List[str]]
    conversation: List[Dict[str, str]]


@dataclass
class Dialogue:
    uid: str
    services: List[str]
    turns: List[Turn]


@dataclass
class ReasoningTurn:
    uid: str
    conversation: List[Dict[str, str]]


@dataclass
class ReasoningDialogue:
    uid: str
    if_target: bool
    turns: List[ReasoningTurn]
    mh_keys: List[str]
    mh_question: str
    mh_answer: str
    sh_question: str
    sh_answer: str


@dataclass
class ReasoningGroupDialogue:
    uid: str
    dialogues: List[ReasoningDialogue]
    target_position: int
    postfix_question: str
    multi_choices: List[str]
    groud_truth: str
    

class MultiwozDataset:
    def __init__(self, dataset_folder_path, propotion=1):
        self.raw_data = self.load_from_dir(dataset_folder_path, propotion)
        self.data = self.build_dataset(self.raw_data)
    
    def load_from_dir(self, dir_pth, propotion):
        """
        Load all the json files from a directory and return a list of json data.
        
        Args:
            propotion (float): the propotion of the data to be loaded. Default is 1.
        """
        concat_data = []
        for root, _, files in os.walk(dir_pth):
            for file in files:
                if file.endswith(".json"):
                    concat_data = concat_data + load_json(os.path.join(root, file))
        return concat_data[:int(len(concat_data) * propotion)]
    
    def remove_inconsistency(self, dataset):
        filtered_dataset = []
        for parsed_dialogue in dataset:
            services = parsed_dialogue.services
            real_services = set()
            for turn in parsed_dialogue.turns:
                real_services.add(turn.service)
            
            if sorted(services) == sorted(list(real_services)):
                filtered_dataset.append(parsed_dialogue)

        return filtered_dataset
    
    def build_dataset(self, data):
        multiwoz = []
        for dialogue_id, dialogue in enumerate(data):
            parsed_turns = []
            for conversation_id in range(0, len(dialogue['turns']), 2):
                turn = dialogue['turns'][conversation_id: conversation_id+2]
                user_turn = turn[0]
                system_turn=turn[1]
                for category in user_turn['frames']:
                    state = category['state']
                    slot_values = state['slot_values']
                    if state['active_intent'] == "NONE":
                        continue
                    active_intent = state['active_intent']
                    service = category['service']
                    break
                
                if len(slot_values.keys()) == 0:
                    continue
                
                parsed_turn = Turn(
                    uid=conversation_id,
                    slot_values=slot_values,
                    active_intent=active_intent,
                    service=service,
                    conversation=[
                        {
                            "role": "user",
                            "content": user_turn['utterance'] 
                        },
                        {
                            "role": "assistant",
                            "content": system_turn['utterance']
                        }
                    ]
                )
                parsed_turns.append(parsed_turn)
            
            if len(dialogue['services']) == 2:
            
                parsed_dialogue = Dialogue(
                    uid=dialogue_id,
                    services=tuple(dialogue['services']),
                    turns=parsed_turns
                )
                multiwoz.append(parsed_dialogue)
        
        multiwoz = self.remove_inconsistency(multiwoz)
        
        return multiwoz
            

class MultihopReasoningQA:
    """
    A class for constructing a dataset for multihop reasoning questions and answers from the MultiWOZ dataset.

    Attributes:
        dataset_folder_path (str): The folder path where the dataset is located.
        propotion (float): The proportion of the data to be used from the dataset.
        target_position (int): The target dialogue's position for reasoning tasks.
        window_size (int): The number of dialogues to consider for each reasoning task.
        mode (str): The mode of reasoning, e.g., 'service' or 'intent'.
        n_options (int): The number of options for multiple-choice questions.

    Methods:
        build_options_pool: Builds a pool of options for multiple-choice questions based on the mode.
        build_multiple_choice: Constructs multiple choice options for a given ground truth.
        build_mh_qa: Generates multi-hop reasoning questions and answers.
        build_sh_qa: Generates single-hop reasoning questions and answers.
        build_dataset: Assembles the dataset for reasoning tasks.
    """
    def __init__(
        self,
        dataset_folder_path,
        propotion,
        target_position,
        window_size,
        mode="service",
        n_options=4,
        seed=42
    ):
        self.dataset_foler_path = dataset_folder_path
        self.target_position = target_position
        self.window_size = window_size
        self.mode = mode
        self.n_options = n_options
        
        random.seed(seed)

        self.multiwoz = MultiwozDataset(dataset_folder_path=dataset_folder_path,propotion=propotion)
        self.options_pool = self.build_options_pool(self.multiwoz, mode)
        self.group_dataset = self.build_dataset()

    def build_options_pool(self, dataset: MultiwozDataset, mode) -> List[str]:
        options_pool = []
        if mode == "service":
            for dialogue in dataset.data:
                options_pool.append(",".join(dialogue.services))
        elif mode == "intent":
            for dialogue in dataset.data:
                continue
            
        elif mode == "inference":
            pass
        else:
            raise NotImplementedError
            
        return list(set(options_pool))

    def build_multiple_choice(self, n_options: int, groud_truth: str, options_pool: List[str]) -> List[str]:
        options_pool.remove(groud_truth)
        options_without_gt = random.sample(options_pool, n_options - 1)
        
        full_options = options_without_gt
        full_options.append(groud_truth)
        random.shuffle(full_options)
        
        options_pool.append(groud_truth)
        
        return full_options
    
    def build_mh_qa(self, dialogue: Dialogue, mode="intent"):
        mh_question = ""
        mh_answer = ""
        
        if mode == "intent":
            mh_question = f"what does user intent to do according to the dialogue {self.target_position}?"
            mh_answer = ",".join(list(set([turn.active_intent for turn in dialogue.turns])))
        elif mode == "service":
            mh_question = f"what are the services the [dialogue {self.target_position}] contain?"
            mh_answer = ",".join(dialogue.services)
        elif mode == "inference":
            mh_question = ""
            mh_answer = ""
            options = ["isn't", "is"]
            answers = ["no", "yes"]
            if len(dialogue.services) != 2:
                mh_question = None
                mh_answer = None
            else:
                services = list(dialogue.services)
                turns_by_service = defaultdict(list)
                for turn in dialogue.turns:
                    turns_by_service[turn.service].append(turn)
                src_service = random.sample(services, 1)[0]
                services.remove(src_service)
                tar_service = services[0]
                
                src_turn = turns_by_service[src_service]
                tar_turn = turns_by_service[tar_service]
                
                src_request = list(src_turn[0].slot_values.values())[0][0] + " " + src_turn[0].service
                
                option_idx = random.randint(0, 1)
                option = options[option_idx]
                
                mh_question = f"Is the statement correct? statement: In the dialogue [{self.target_position}], the user request for a {src_request}, another topic user ask for {option} {tar_service}. Answer(yes/no):"
                
                mh_answer = answers[option_idx]
            
        else:
            raise NotImplementedError
        
        return mh_question, mh_answer
    
    def build_sh_qa(self, dialogue: Dialogue):
        sh_question = ""
        sh_answer = ""
        
        return sh_question, sh_answer

    def build_dataset(self):
        window_size = self.window_size
        position = self.target_position
        dataset = self.multiwoz.data
        group_dataset = []
        for group_id in range(0, len(dataset), window_size):
            if group_id + window_size > len(dataset):
                continue
            group_dialogues: List[Dialogue] = dataset[group_id: group_id+window_size]
            group_reasoning_dialogues = []
            for dialogue_id, dialogue in enumerate(group_dialogues):
                reasoning_turns = []
                for turn in dialogue.turns:
                    reasoning_turns.append(ReasoningTurn(
                        uid=turn.uid,
                        conversation=turn.conversation
                    ))
                
                mh_question, mh_answer = self.build_mh_qa(dialogue, mode=self.mode)
                sh_question, sh_answer = self.build_sh_qa(dialogue)    
                
                
                group_reasoning_dialogues.append(ReasoningDialogue(
                    uid=dialogue_id,
                    turns=reasoning_turns,
                    mh_keys=None,
                    mh_question=mh_question,
                    mh_answer=mh_answer,
                    sh_question=sh_question,
                    sh_answer=sh_answer,
                    if_target= str(dialogue.uid) == str(position)
                ))
            

            postfix_question = group_reasoning_dialogues[self.target_position].mh_question
            groud_truth = group_reasoning_dialogues[self.target_position].mh_answer
            if self.mode != "inference":
                multi_choices = self.build_multiple_choice(self.n_options, groud_truth, self.options_pool)
            else:
                multi_choices = ["no", "yes"]
            
            
            group_dataset.append(ReasoningGroupDialogue(
                uid=group_id,
                dialogues=group_reasoning_dialogues,
                target_position=self.target_position,
                postfix_question=postfix_question,
                multi_choices=multi_choices,
                groud_truth=groud_truth
            ))
        
        return group_dataset
    
    def save_as_json(self, root_path, mode="inference"):
        formatted_dataset = []
        options_and_gts = []
        for grouped_dialogue in self.group_dataset:
            formatted_group_dialogue = []

            for dialogue in grouped_dialogue.dialogues:
                for turn in dialogue.turns:
                    turn.conversation[0]['content'] = f"[dialogue {dialogue.uid}] " + turn.conversation[0]['content']
                    formatted_group_dialogue.extend(turn.conversation)
            
            target_dialogue = grouped_dialogue.dialogues[self.target_position]
            
            formatted_group_dialogue.append({
                "role": "user",
                "content": target_dialogue.mh_question
            })
            
            options_and_gts.append({
                "options": grouped_dialogue.multi_choices,
                "gt": target_dialogue.mh_answer
            })
            
            formatted_dataset.append(formatted_group_dialogue)
        
        with open(os.path.join(root_path, f"multiwoz-{self.window_size}-{self.target_position}.json"), 'w') as f:
            json.dump(formatted_dataset, f, indent=4)
            
        with open(os.path.join(os.path.join(root_path, "labels"), f"multiwoz-{self.window_size}-{self.target_position}-label.json"), 'w') as f:
            json.dump(options_and_gts, f, indent=4)

