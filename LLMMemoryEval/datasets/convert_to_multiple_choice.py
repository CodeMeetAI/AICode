import os
import json

class MultipleChoiceDataset:
    def __init__(
        self, 
        data_dir,
        save_dir,
        question = None,
        window_size = 3,
        target_position = 0
    ) -> None:
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.window_size = window_size
        self.raw_data = self.load_dataset()
        if question is None:
            self.question = "According to the context, what's the first question that {user_id} asked? \nHint: Only reply the correct option character (i.e., '(<Option>)'). \nOption: "
            # self.question = "According to the context, what's the first question that {user_id} asked? Choose from the options: "
        else:
            self.question = question
            
        self.chat = self.grouping_sample(window_size=window_size, target_position=target_position)
        
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
    
    def add_options(self, labels):
        """_summary_

        Args:
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        labels_with_options = [f"({chr(65 + i)}). {item}" for i, item in enumerate(labels)]
        return ' '.join(labels_with_options)
    
    def grouping_sample(self, window_size, target_position = 0):
        """_summary_

        Args:
            window_size (_type_): _description_
            target_position (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        grouped_dataset = []
        
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

            multiple_choice_str = self.add_options(grouped_labels)
            
            question = self.question.format(user_id=target_user_id) + multiple_choice_str + "\nAnswer:"
            

            
            user_query_dict = {
                "role": "user",
                "content": question
            }
            grouped_conversations.append(user_query_dict)
            grouped_dataset.append(grouped_conversations)
        
        return grouped_dataset

    def save_json(self, file_name):
        save_dir = os.path.join(self.save_dir, file_name)
        with open(save_dir, "w") as f:
            json.dump(self.chat, f ,indent=4)


if __name__ == "__main__":
    data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/MultiWOZ_2.2/multiwoz_2.2.json"
    save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/MultiWOZ_2.2/"
    # data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/frames/frames.json"
    # save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/frames/"
    # data_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/nq_dialogues_gemma.json"
    # save_dir = "/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/"
    
    window_len = 6
    multiple_choice_dataset = MultipleChoiceDataset(data_dir=data_dir, save_dir=save_dir, window_size=window_len)
    multiple_choice_dataset.save_json(file_name = f"multiwoz_grouped_{window_len}.json")
    # multiple_choice_dataset.save_json(file_name = f"frames_grouped_{window_len}.json")
    # multiple_choice_dataset.save_json(file_name = "natural_questions_grouped.json")

            
    
        
