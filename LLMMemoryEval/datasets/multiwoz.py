import json
import os


## Version 2

class MultiwozDataLoader:
    def __init__(self, pth: str, model: str):
        self.pth = pth
        self.data = {}
        self.model = model
        self.background = {
            "lamma2": "<<SYS>>All the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<</SYS>>",
            "gemma": "<start_of_turn>system\nAll the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<end_of_turn>\n"
        }
        self.templates = {
            "lamma2": [
                    "<s>[INST]{content}[/INST]", # User prompt
                    "{content}</s>", # Model answer
                    "Hi, I'm EJ{num}. {content}" # The first sentence in the conversation
                ],
            "gemma": [
                    "<start_of_turn>user\n{content}<end_of_turn>\n", # The prompt temp. for user
                    "<start_of_turn>model\n{content}<end_of_turn>\n", # The prompt temp. for model
                    "Hi, I'm EJ{num}. {content}" # The first sentence in the conversation
                ]
                        }
        self.num = 1
        
    def load_processed_data(self):
        if os.path.exists(full_pth := os.path.join(self.pth, "multiwoz_2.2.json")):
            with open(full_pth, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
        
    def sample_select(self, n: int):
        """
            Merge n conversations
        """
        data = self.data["train"][:1000]  # + self.data["test"] + self.data["dev"]
        conversations = []
        questions     = []
        tmp = self.background["gemma"]
        stop_point = 1
        for i, conversation in enumerate(data):
            tmp += conversation[1]
            questions.append(conversation[1].split("\n")[1])
            if i % n == 0 and i > 0:
                numbers = list(range(stop_point, i+1))    #list(range(i-n if n//5 >0 else i-1, i, n//5 if n//5 >0 else 1))
                # random.shuffle(numbers)
                stop_point += n
                question = ""
                for j in numbers:
                    question += f"What's the question that EJ{str(j+1).zfill(6)} asked? "
                tmp += self.templates[self.model][1].format(content="According to the conversation above - " + question) + "<start_of_turn>model\n"
                conversations.append(tmp)
                tmp = self.background["gemma"]
        with open(f"processed_data_{n}.json", "w") as f:
            json.dump(conversations, f)

        with open(f"answer_{n}.json", "w") as f:
            json.dump(questions, f)
    
    def load_data(self):
        """
            Get the data from 3 parts (train, dev, test), and store in the self.data
        """
        for dirname in ["train", "test", "dev"]:
            self.load_from_dir(dirname)
        with open(os.path.join(self.pth, "multiwoz_2.2.json"), "w") as f:
            json.dump(self.data, f)
    
    def load_from_dir(self, dir_pth: str):
        """
            Given the directory name and get the data.
            
            Args:
                dir_pth: the name of directory. For example, "train", "test", and "dev"
        """
        for file in os.listdir(dirname := os.path.join("/home/eidf018/eidf018/s2484588-epcc/MLP/Dataset/MultiWOZ_2.2", dir_pth)):
            if file.endswith(".json"):
                self.parse_data(os.path.join(dirname, file), dirname)
                    
    def parse_data(self, filename: str, dirname: str):
        """
            Parsing a certain json file.
            
            Args:
                filename: The name of the file, end with ".json"
                dir_pth: the name of directory.
        """
        with open(filename, "r") as f:
            dic = json.load(f)
            dir_point = self.data.get(dirname.split("/")[-1], [])
            dir_point = dir_point + self.filter_the_info(dic)
        self.data[dirname.split("/")[-1]] = dir_point
    
    def filter_the_info(self, conversations):
        samples = []
        for conversation in conversations:
            sample = {"user_id": None, "conversations": []}
            for turn in conversation["turns"]:
                sample["conversations"].append(turn["utterance"])
            sample["user_id"] = str(self.num).zfill(6)
            # sample[0][0] = True
            # sample_conv = ""
            # sample_conv += self.templates[-1].format(**{"num": sample["user_id"], "content": sample[1][0]})
            self.num += 1
            sample["conversations"][0] = self.templates[self.model][-1].format(num=self.num, content=turn)
            for idx, turn in enumerate(sample["conversations"]):
                sample["conversations"][idx] = self.templates[self.model][idx % 2].format(content=turn)
            # sample[1] = sample_conv
            samples.append(sample)
        return samples
    
    def find_active(self, turns):
        for turn in turns:
            try:
                if turn["state"]["active_intent"] != "NONE":
                    return turn["state"]["active_intent"]
            except:
                return None
        return None

# class MultiwozDataLoader:
#     def __init__(self, pth: str):
#         self.pth = pth
#         self.data = {}
#         self.prefix = ["<start_of_turn>model\n{content}<end_of_turn>\n", "<start_of_turn>user\n{content}<end_of_turn>\n", "<start_of_turn>user\nHi, I'm EJ{num}. {content}<end_of_turn>\n"]
#         self.num = 1
        
#     def load_processed_data(self):
#         if os.path.exists(full_pth := os.path.join(self.pth, "multiwoz_2.2.json")):
#             with open(full_pth, "r") as f:
#                 self.data = json.load(f)
#         else:
#             self.load_data()
        
#     def sample_select(self, n: int):
#         """
#             Merge n conversations
#         """
#         data = self.data["train"][:1000]# + self.data["test"] + self.data["dev"]
#         conversations = []
#         questions     = []
#         tmp = "<start_of_turn>system\nAll the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<end_of_turn>\n"
#         stop_point = 1
#         for i, conversation in enumerate(data):
#             tmp += conversation[1]
#             questions.append(conversation[1].split("\n")[1])
#             if i % n == 0 and i>0:
#                 numbers = list(range(stop_point, i+1))    #list(range(i-n if n//5 >0 else i-1, i, n//5 if n//5 >0 else 1))
#                 # random.shuffle(numbers)
#                 stop_point += n
#                 question = ""
#                 for j in numbers:
#                     question += f"What's the question that EJ{str(j+1).zfill(6)} asked? "
#                 tmp += self.prefix[1].format(content="According to the conversation above - " + question) + "<start_of_turn>model\n"
#                 conversations.append(tmp)
#                 tmp = "<start_of_turn>system\nAll the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<end_of_turn>\n"
#         with open(f"processed_data_{n}.json", "w") as f:
#             json.dump(conversations, f)

#         with open(f"answer_{n}.json", "w") as f:
#             json.dump(questions, f)
    
#     def load_data(self):
#         """
#             Get the data from 3 parts (train, dev, test), and store in the self.data
#         """
#         for dirname in ["train", "test", "dev"]:
#             self.load_from_dir(dirname)
#         with open(os.path.join(self.pth, "multiwoz_2.2.json"), "w") as f:
#             json.dump(self.data, f)
    
#     def load_from_dir(self, dir_pth: str):
#         """
#             Given the directory name and get the data.
            
#             Args:
#                 dir_pth: the name of directory. For example, "train", "test", and "dev"
#         """
#         cnt = 0
#         for file in os.listdir(dirname := os.path.join("/home/eidf018/eidf018/s2484588-epcc/MLP/Dataset/MultiWOZ_2.2", dir_pth)):
#             if file.endswith(".json"):
#                 self.parse_data(os.path.join(dirname, file), dirname)
#             cnt += 1
#             if cnt > 3:
#                 break
                    
#     def parse_data(self, filename: str, dirname: str):
#         """
#             Parsing a certain json file.
            
#             Args:
#                 filename: The name of the file, end with ".json"
#                 dir_pth: the name of directory.
#         """
#         with open(filename, "r") as f:
#             dic = json.load(f)
#             dir_point = self.data.get(dirname.split("/")[-1], [])
#             dir_point = dir_point + self.filter_the_info(dic)
#         self.data[dirname.split("/")[-1]] = self.data.get(dirname.split("/")[-1], []) + dir_point
    
#     def filter_the_info(self, conversations):
#         samples = []
#         for conversation in conversations:
#             sample = [[], []]
#             for turn in conversation["turns"]:
#                 sample[0].append(False)
#                 # sample[0].append(self.find_active(turn["frames"]))
#                 sample[1].append(turn["utterance"])
#             sample[0][0] = True
#             sample_conv = ""
#             sample_conv += self.prefix[-1].format(**{"num": str(self.num).zfill(6), "content": sample[1][0]})
#             self.num += 1
#             for idx, turn in enumerate(sample[1][1:]):
#                 sample_conv += self.prefix[idx%2].format(**{"content": turn})
#             sample[1] = sample_conv
#             samples.append(sample)
#         return samples
    
## Version 2

class MultiwozDataLoader:
    def __init__(self, pth: str, model: str):
        self.pth = pth
        self.data = {}
        self.model = model
        self.templates = {
            "lamma2": [
                    "",
                    "",
                    "Hi, I'm EJ{num}. {content}" # The first sentence in the conversation
                ],
            "gemma": [
                    "<start_of_turn>model\n{content}<end_of_turn>\n", # The prompt temp. for model
                    "<start_of_turn>user\n{content}<end_of_turn>\n", # The prompt temp. for user
                    "Hi, I'm EJ{num}. {content}" # The first sentence in the conversation
                ]
                        }
        self.num = 1
        
    def load_processed_data(self):
        if os.path.exists(full_pth := os.path.join(self.pth, "multiwoz_2.2.json")):
            with open(full_pth, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
        
    def sample_select(self, n: int):
        """
            Merge n conversations
        """
        data = self.data["train"][:1000]# + self.data["test"] + self.data["dev"]
        conversations = []
        questions     = []
        tmp = "<start_of_turn>system\nAll the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<end_of_turn>\n"
        stop_point = 1
        for i, conversation in enumerate(data):
            tmp += conversation[1]
            questions.append(conversation[1].split("\n")[1])
            if i % n == 0 and i > 0:
                numbers = list(range(stop_point, i+1))    #list(range(i-n if n//5 >0 else i-1, i, n//5 if n//5 >0 else 1))
                # random.shuffle(numbers)
                stop_point += n
                question = ""
                for j in numbers:
                    question += f"What's the question that EJ{str(j+1).zfill(6)} asked? "
                tmp += self.templates[self.model][1].format(content="According to the conversation above - " + question) + "<start_of_turn>model\n"
                conversations.append(tmp)
                tmp = "<start_of_turn>system\nAll the question should be answered as brief as possible - even only keywords! And don't talk about other information that the user doesn't ask.<end_of_turn>\n"
        with open(f"processed_data_{n}.json", "w") as f:
            json.dump(conversations, f)

        with open(f"answer_{n}.json", "w") as f:
            json.dump(questions, f)
    
    def load_data(self):
        """
            Get the data from 3 parts (train, dev, test), and store in the self.data
        """
        for dirname in ["train", "test", "dev"]:
            self.load_from_dir(dirname)
        with open(os.path.join(self.pth, "multiwoz_2.2.json"), "w") as f:
            json.dump(self.data, f)
    
    def load_from_dir(self, dir_pth: str):
        """
            Given the directory name and get the data.
            
            Args:
                dir_pth: the name of directory. For example, "train", "test", and "dev"
        """
        for file in os.listdir(dirname := os.path.join("/home/eidf018/eidf018/s2484588-epcc/MLP/Dataset/MultiWOZ_2.2", dir_pth)):
            if file.endswith(".json"):
                self.parse_data(os.path.join(dirname, file), dirname)
                    
    def parse_data(self, filename: str, dirname: str):
        """
            Parsing a certain json file.
            
            Args:
                filename: The name of the file, end with ".json"
                dir_pth: the name of directory.
        """
        with open(filename, "r") as f:
            dic = json.load(f)
            dir_point = self.data.get(dirname.split("/")[-1], [])
            dir_point = dir_point + self.filter_the_info(dic)
        self.data[dirname.split("/")[-1]] = dir_point
    
    def filter_the_info(self, conversations):
        samples = []
        for conversation in conversations:
            sample = {"user_id": None, "conversations": []}
            for turn in conversation["turns"]:
                sample["conversations"].append(turn["utterance"])
            sample["user_id"] = str(self.num).zfill(6)
            # sample[0][0] = True
            # sample_conv = ""
            # sample_conv += self.templates[-1].format(**{"num": sample["user_id"], "content": sample[1][0]})
            self.num += 1
            sample["conversations"][0] = self.templates[self.model][-1].format(num=self.num, content=turn)
            for idx, turn in enumerate(sample["conversations"]):
                sample["conversations"][idx] = self.templates[self.model][idx%2].format(content=turn)
            # sample[1] = sample_conv
            samples.append(sample)
        return samples
    
    def find_active(self, turns):
        for turn in turns:
            try:
                if turn["state"]["active_intent"] != "NONE":
                    return turn["state"]["active_intent"]
            except:
                return None
        return None