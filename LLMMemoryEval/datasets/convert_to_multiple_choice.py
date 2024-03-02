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
        self.questions = self.get_questions()
        self.n_labels = n_labels
        self.labels = None
        
        if question is None:
            self.question = "According to the context, what's the first question that {user_id} asked?"
            # self.question = "According to the context, what's the first question that {user_id} asked? Choose from the options: "
        else:
            self.question = question
            
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
    
    def insert_user_id(self, conversations, user_id):
        conversations[0]['content'] = f"Hi, I'm {user_id}, "  + conversations[0]['content']
        return conversations
    
    def pad_truncate_labels(self, labels, gt):
        if len(labels) > self.n_labels - 1:
            labels = labels[:self.n_labels - 1]
        else:
            # self.questions store all questions(conversation's first question)
            # remove the gt from self.questions and then random select questions for padding
            questions = copy.deepcopy(self.questions)
            questions.remove(gt)
            pad_question = random.sample(questions, self.n_labels - 1 - len(labels))
            labels += pad_question

        return labels
    
    def add_options(self, labels):
        """_summary_

        Args:
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        labels_with_options = [f"({chr(65 + i)}): {item}" for i, item in enumerate(labels)]
        return ' '.join(labels_with_options)
    
    def get_questions(self):
        questions = []
        for sample in self.raw_data:
            questions.append(sample['label'])
        
        return questions
    
    def grouping_sample(self, window_size, target_position = 0):
        """_summary_

        Args:
            window_size (_type_): _description_
            target_position (int, optional): The position of correct answer we choose. Defaults to 0 (the first person).

        Returns:
            _type_: _description_
        """
        grouped_dataset = []
        ground_truths = []
        
        for index in range(0, len(self.raw_data), window_size):

            bulk_sample_dict = self.raw_data[index: index + window_size]
            if len(bulk_sample_dict) < window_size:
                continue
            
            grouped_conversations = []
            grouped_labels = []
            
            target_user_id = bulk_sample_dict[target_position]["user_id"]
            
            for single_sample_dict in bulk_sample_dict:
                user_id = single_sample_dict['user_id']
                
                conversations = self.insert_user_id(single_sample_dict['conversations'], user_id)
                if conversations[-1]['role'] == "user":
                    conversations = conversations[:-1]
                grouped_conversations.extend(conversations)
                grouped_labels.append(single_sample_dict['label'])
        
            label = grouped_labels[target_position]
            
            grouped_labels.remove(label)
            processed_labels = self.pad_truncate_labels(grouped_labels, gt=label)
            processed_labels.append(label)
            
            random.shuffle(processed_labels)
            
            multiple_choice_str = self.add_options(processed_labels)
            ground_truths.append(chr(65+processed_labels.index(label)))
                
            
            question = self.question.format(user_id=target_user_id).strip() + multiple_choice_str + "\nAnswer:<only one character of A to D>"
            
            user_query_dict = {
                "role": "user",
                "content": question
            }
            grouped_conversations.append(user_query_dict)
            grouped_dataset.append(grouped_conversations)
        
        return grouped_dataset, ground_truths

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
    
    window_lens = [2,3,4,5,6,7,8]
    target_positions = [0, 1, 2]
    for window_len in window_lens:
        for target_position in target_positions:
            multiple_choice_dataset = MultipleChoiceDataset(data_dir=data_dir, save_dir=save_dir, window_size=window_len, target_position=target_position)
            # multiple_choice_dataset.save_json(file_name = f"multiwoz_grouped_{window_len}")
            # multiple_choice_dataset.save_json(file_name = f"frames_grouped_{window_len}")
            multiple_choice_dataset.save_json(file_name = f"natural_questions_grouped_{window_len}")
            
            print("win_size: {}, n_sample: {}, n_labels: {}".format(window_len, len(multiple_choice_dataset.chat), len(multiple_choice_dataset.labels)))
            
    
        
