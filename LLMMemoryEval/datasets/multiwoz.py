import json
import re
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
    turns_count: int


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
    ground_truth: str
    

class MultiwozDataset:
    def __init__(self, dataset_folder_path, propotion=1):
        random.seed(42)
        self.raw_data = self.load_from_dir(dataset_folder_path, propotion)
        self.data = self.build_dataset(self.raw_data)
        self.data = self.delete_topics()
        for i in range(5):
            random.shuffle(self.data)
    
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
            
            if len(dialogue['services']) == 2 and len(parsed_turns) >= 0 and len(parsed_turns) < 100:
            
                parsed_dialogue = Dialogue(
                    uid=dialogue_id,
                    services=tuple(dialogue['services']),
                    turns=parsed_turns,
                    turns_count=len(parsed_turns)
                )
                multiwoz.append(parsed_dialogue)
        
        multiwoz = self.remove_inconsistency(multiwoz)
        
        return multiwoz
        
    def delete_topics(self):
        multiwoz = []
        for dialogue in self.data:
            if "taxi" not in dialogue.services and "bus" not in dialogue.services:
                multiwoz.append(dialogue)
            
        return multiwoz

    def get_target_dialogues(self, n_target_dialogue):
        return random.sample(self.data, n_target_dialogue)
            

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
        n_options=3,
        seed=42
    ):
        self.dataset_foler_path = dataset_folder_path
        self.target_position = target_position
        self.window_size = window_size
        self.mode = mode
        self.n_options = n_options
        self.appeared_options = []
        
        random.seed(seed)

        self.multiwoz = MultiwozDataset(dataset_folder_path=dataset_folder_path,propotion=propotion)
        self.options_pool = self.build_options_pool(self.multiwoz, mode)
        
        n_target_dialogue = len(self.multiwoz.data) // 5
        self.target_dialogues = self.multiwoz.get_target_dialogues(n_target_dialogue=n_target_dialogue)
        
        self.group_dataset = self.build_dataset()

    def get_all_slot_values_key(self):
        all_keys = defaultdict(int)
        for dialogue in self.multiwoz.data:
            for turn in dialogue.turns:
                for key in turn.slot_values.keys():
                    all_keys[key] += 1
        
        return all_keys
    def build_options_pool(self, dataset: MultiwozDataset, mode) -> List[str]:
        options_pool = []
        if mode == "service":
            for dialogue in dataset.data:
                options_pool.append(",".join(dialogue.services))
        elif mode == "intent":
            for dialogue in dataset.data:
                continue
            
        elif mode == "inference":
            for dialogue in dataset.data:
                options_pool.extend(dialogue.services)
        
        else:
            raise NotImplementedError
            
        return list(set(options_pool))

    def build_multiple_choice(self, n_options: int, ground_truth: str, options_pool: List[str], appeared_option=None) -> List[str]:
        options_pool.remove(ground_truth)
        if appeared_option is not None:
            options_pool.remove(appeared_option)
        options_without_gt = random.sample(options_pool, n_options - 1)
        
        full_options = options_without_gt
        full_options.append(ground_truth)
        random.shuffle(full_options)
        
        options_pool.append(ground_truth)
        if appeared_option is not None:
            options_pool.append(appeared_option)
        
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
                src_request = src_request.strip()
                
                # mh_question = f"Is the statement correct? statement: In the dialogue [{self.target_position}], the user request for a {src_request}, another topic user ask for {option} {tar_service}. Answer(yes/no):"
                mh_question = f"Question: There are two topics in dialogue {self.target_position}, one topic is '{src_turn[0].service}'. What's the another topic?\n Options: " + "{options}\n Your choice: "
                mh_answer = tar_service
            
        else:
            raise NotImplementedError
        
        return mh_question, mh_answer
    
    def build_sh_qa(self, dialogue: Dialogue):
        sh_question = ""
        sh_answer = ""
        
        return sh_question, sh_answer

    def build_dataset(self):
        
        window_size = self.window_size
        group_dataset = []
        all_dialogue = self.multiwoz.data
        
        prefixes = ['(A).', '(B).', '(C).']
        
        all_context_dialogue = remove_elements(all_dialogue, self.target_dialogues)
        for group_id, target_dialogue in enumerate(self.target_dialogues):
            group_dialogues, all_context_dialogue = sample_and_remove(all_context_dialogue, window_size - 1)
            group_dialogues.insert(self.target_position, target_dialogue)
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
                    if_target= str(dialogue.uid) == str(self.target_position)
                ))
            

            postfix_question = group_reasoning_dialogues[self.target_position].mh_question
            ground_truth = group_reasoning_dialogues[self.target_position].mh_answer
            if self.mode == "inference":
                appeared_option = re.search(r"'(.*?)'", postfix_question).group(1)
                multi_choices = self.build_multiple_choice(self.n_options, ground_truth, self.options_pool, appeared_option=appeared_option)
                ground_truth_option = prefixes[multi_choices.index(ground_truth)] + ground_truth
                multi_choices_str = " ".join(f"{prefixes[i]} {option}" for i, option in enumerate(multi_choices))
                self.appeared_options.append(appeared_option)
                
            else:
                multi_choices = self.build_multiple_choice(self.n_options, ground_truth, self.options_pool)

            dialogue_grouped_with_options = []
            for dialogue in group_reasoning_dialogues:
                dialogue.mh_question = dialogue.mh_question.format(options=multi_choices_str)
                dialogue_grouped_with_options.append(dialogue)
            
            group_dataset.append(ReasoningGroupDialogue(
                uid=group_id,
                dialogues=dialogue_grouped_with_options,
                target_position=self.target_position,
                postfix_question=postfix_question,
                multi_choices=[f"{prefixes[i]}{option}" for i, option in enumerate(multi_choices)],
                ground_truth=ground_truth_option
            ))
        
        assert len(self.appeared_options) == len(group_dataset), "Mismatch"
        
        return group_dataset
    
    def save_as_json(self, root_path, mode="inference"):
        formatted_dataset = []
        options_and_gts = []
        for i, grouped_dialogue in enumerate(self.group_dataset):
            formatted_group_dialogue = []

            for j, dialogue in enumerate(grouped_dialogue.dialogues):
                dialogue.turns[0].conversation[0]["content"] = f"dialogue {j} started.\n" + dialogue.turns[0].conversation[0]['content']
                for turn in dialogue.turns:
                    formatted_group_dialogue.extend(turn.conversation)

                
            target_dialogue = grouped_dialogue.dialogues[self.target_position]
            
            formatted_group_dialogue.append({
                "role": "user",
                "content": target_dialogue.mh_question
            })
            
            options_and_gts.append({
                "options": grouped_dialogue.multi_choices,
                "gt": grouped_dialogue.ground_truth,
            })
            
            formatted_dataset.append(formatted_group_dialogue)
        
        with open(os.path.join(root_path, f"multiwoz-{self.window_size}-{self.target_position}-{self.mode}-new.json"), 'w') as f:
            json.dump(formatted_dataset, f, indent=4)
            
        with open(os.path.join(os.path.join(root_path, "labels"), f"multiwoz-{self.window_size}-{self.target_position}-label-{self.mode}-new.json"), 'w') as f:
            json.dump(options_and_gts, f, indent=4)


def sample_and_remove(lst, k):
    
    selected_indices = random.sample(range(len(lst)), k)
    
    sampled_elements = [lst[i] for i in selected_indices]
    
    for i in sorted(selected_indices, reverse=True):
        del lst[i]
    
    return sampled_elements, lst

def remove_elements(lst, elements_to_remove):
    
    updated_list = [item for item in lst if item not in elements_to_remove]
    
    return updated_list