import json
import random

class NaturalQuestionsDataLoader():
    def __init__(self, file_path) -> None:
        self.data = self.load_data(file_path)
        self.user_id_format = "EJ{num:05d}"

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def generate_dialogues(self, num_questions_per_dialogue, num_dialogues):
        dialogues = []
        questions_indices = list(range(len(self.data)))

        for i in range(num_dialogues):
            user_id = self.user_id_format.format(num=i+1)
            conversation = []
            # 随机选择指定数量的问题
            selected_indices = random.sample(questions_indices, num_questions_per_dialogue)
            label = ""
            for j, idx in enumerate(selected_indices):
                question = self.data[idx]['question']
                # 记录第一个问题为label
                if j == 0:
                    label = question
                answer = self.data[idx]['short_answers']
                conversation.append({"role": "assistant", "content": question})
                conversation.append({"role": "user", "content": answer})
                # conversation.append({"role": "user", "content": question})
                # conversation.append({"role": "model", "content": answer})
            dialogues.append({"user_id": user_id, "conversations": conversation,"label":label})

        return dialogues
    
if __name__ == "__main__":
    file_path = '/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/extracted_short_answers.json'  
    data_loader = NaturalQuestionsDataLoader(file_path)
    num_questions_per_dialogue = 5  # 每轮对话的问题数
    num_dialogues = 2000  # 生成对话的数量
    dialogues = data_loader.generate_dialogues(num_questions_per_dialogue, num_dialogues)

    output_file_path = '/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data/natural_questions/nq_dialogues_gemma.json'  
    with open(output_file_path, 'w') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)