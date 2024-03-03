import os
import random
import copy
import json

class MultipleChoiceDataset:
    def __init__(
        self, 
        data_dir,
        save_dir,
        question = None,
        window_size = 3,
        target_position = 0,
        n_labels = 4
    ) -> None:
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.window_size = window_size
        self.raw_data = self.load_dataset()
        self.position = target_position
        
        self.prefix = "[turn {index}] "
        self.postfix = "What does user say in [turn {index}]? Answer:"
        self.conversations = self.get_conversations()
        self.conversations_content = self.get_conversations_content()
        
        self.n_labels = n_labels
        self.labels = None
            
        if target_position == 0: # choose the first person
            self.chat, self.labels = self.grouping_sample(window_size=window_size, target_position=target_position)
        elif target_position == 1: # choose the middle person
            self.chat, self.labels = self.grouping_sample(window_size=window_size, target_position=window_size//2)
        else:
            self.chat, self.labels = self.grouping_sample(window_size=window_size, target_position=window_size-1)
        
    def load_dataset(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with open(self.data_dir, "r") as f:
            raw_data = json.load(f)
        return raw_data
    
    def construct_multiple_choices(self, ground_truth):
        conversations = self.conversations_content
        conversations.remove(ground_truth)
        random_sampled_choices = random.sample(conversations, self.n_labels - 1)
        random_sampled_choices.append(ground_truth)
        
        random.shuffle(random_sampled_choices)
        conversations.append(ground_truth)
        return random_sampled_choices
    
    def add_options(self, labels):
        """_summary_

        Args:
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        labels_with_options = [f"({chr(65 + i)}): {item}" for i, item in enumerate(labels)]
        return ' '.join(labels_with_options)
    
    def get_conversations_content(self):
        conversation_content = []
        for conv in self.conversations:
            conversation_content.append(conv['content'])
                
        return conversation_content
    
    def get_conversations(self):
        conversations = []
        last_role = None
        for sample in self.raw_data:
            for i, conv in enumerate(sample['conversations']):
                if conv['role'] != last_role:
                    conversations.append(conv)
                last_role = sample['conversations'][i % len(sample['conversations'])]['role']
        return conversations
    
    def grouping_sample(self, window_size, target_position = 0):
        """_summary_

        Args:
            window_size (_type_): _description_
            target_position (int, optional): The position of correct answer we choose. Defaults to 0 (the first person).

        Returns:
            _type_: _description_
        """
        grouped_dataset = []
        options = []
        
        conversations = self.conversations
        
        
        # assert len(conversations) % 2 == 0, "n user input != n model output, exsit incomplete conversations"
        
        last_role = None
        
        for conv_index in range(0, len(conversations), window_size * 2):
            
            conversations_per_group = conversations[conv_index: conv_index + window_size * 2]
            
            if conversations_per_group[-1]['role'] == 'user':
                conversations_per_group = conversations_per_group[:-1]
            
            if len(conversations_per_group) < window_size * 2:
                continue
            
            target_user_input_per_group = conversations_per_group[target_position * 2] # User's input
            # target_assistant_output_per_group = conversations_per_group[target_position * 2 + 1] # Assistant's input
            multiple_choice = self.construct_multiple_choices(target_user_input_per_group['content'])

            options.append({
                "choices": multiple_choice,
                "gt": target_user_input_per_group['content']
            })
            
            
            
            for group_conv_index in range(0, len(conversations_per_group), 2):
                
                self.prefix.format(index=int(group_conv_index / 2)) + conversations_per_group[group_conv_index]['content']

            
            conversations_per_group.append({
                "role": "user",
                "content": self.postfix.format(index=target_position)
            })
            
            grouped_dataset.append(conversations_per_group)
        
            if len(grouped_dataset) >= 1000:
                break
        
        return grouped_dataset, options

    def save_json(self, file_name):
        pos = {0: "_first", 1: "_middle", 2:"_last"}
        save_dir = os.path.join(self.save_dir, file_name) + pos[self.position] + ".json"
        with open(save_dir, "w") as f:
            json.dump(self.chat, f ,indent=4)
        
        save_dir = os.path.join(os.path.join(self.save_dir, "labels"), file_name) + pos[self.position] + "-label.json"
        with open(save_dir, "w") as f:
            json.dump(self.labels, f ,indent=4)


if __name__ == "__main__":
    random.seed(42)
    # data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/multiwoz/multiwoz_2.2.json"
    # save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/multiwoz/"
    # data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/frames/frames.json"
    # save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/frames/"
    data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/nq_dialogues.json"
    save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/"
    
    window_lens = [4,5,6,7,8]
    target_positions = [0, 1, 2]
    for window_len in window_lens:
        for target_position in target_positions:
            multiple_choice_dataset = MultipleChoiceDataset(data_dir=data_dir, save_dir=save_dir, window_size=window_len, target_position=target_position)
            # multiple_choice_dataset.save_json(file_name = f"multiwoz_grouped_new_{window_len}")
            # multiple_choice_dataset.save_json(file_name = f"frames_grouped_new_{window_len}")
            multiple_choice_dataset.save_json(file_name = f"natural_questions_grouped_new_{window_len}")
            
            print("win_size: {}, n_sample: {}, n_labels: {}".format(window_len, len(multiple_choice_dataset.chat), len(multiple_choice_dataset.labels)))
            
    
        
