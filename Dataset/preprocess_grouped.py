import json
import os
import pickle

dataset = "MultiWOZ_2.2"


def find_active(frame):
    for i in range(len(frame)):
        if (intent := frame[i]).get("state", {"active_intent": "NONE"})["active_intent"] != "NONE":
            return intent["service"]

class Interviewer:
    def __init__(self):
        self.prefix = "Now you should act like server in a story world, and I will ask you some question about this world. You can use the real world information to makeup any information in this story world (remember! you are not just a robot, but a server, and if you don't know the answer, just briefly tell me what you said before in this format: 'Answer: <you answer>' )."
        self.q = {
            "hotel": "The name of hotel you mentioned is: ",
            "restaurant": "The name of restaurant you mentioned is: ",
            "attraction": "The name of attraction you mentioned is: ",
            "taxi": "The departure for the taxi is: ",
            "train": "The leaveat of the train is: ",
            "bus": "The destination of the bus is: ",
            "police": "The police postcode/address/phone_number is: ",
            "hospital": "The hospital postcode/address/phone_number is: ",
            # "hotel": "hotel name is: [The name of hotel]",
            # "restaurant": "restaurant name is: [The name of restaurant]",
            # "attraction": "attraction name is: [The name of attraction]",
            # "taxi": "departure for the taxi is: [The departure of taxi]",
            # "train": "leaveat of the train is: [The leaveat of the train]",
            # "bus": "destination of the bus is: [The destination of the bus]",
            # "police": "police postcode/address/phone_number is: [The police postcode or address or phone number]",
            # "hospital": "hospital postcode/address/phone_number is: [The hospital postcode or address or phone number]",
        }

    def __call__(self, service):
        return self.q[service] + self.prefix


samples = []
cur_spl = None
interviewer = Interviewer()

for dir_name in ["train", "dev", "test"]:
    dir_path = os.path.join(dataset, dir_name)
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            with open(os.path.join(dir_path, file), "r") as f:
                data = json.load(f)
        for i in range(len(data)):
            samples.append(([f"Now you should act like server in a story world, and I will ask you some question about this world. You can use the real world information to makeup any information in this story world (remember! you are not just a robot, but a server). I am the {i+1}-th person. "], [None]))
            for d in range(0, len(data[i]["turns"]), 2):
                # print(dialogue["frames"])
                dialogue = data[i]["turns"][d]
                service  = find_active(dialogue["frames"])
                if service:
                    cur_spl = service
                if len(old := samples[-1][1]) and old[-1] and (service != old[-1]):
                    # if old[-1]:
                    samples[-1][0].append(interviewer(old[-1]))
                    samples[-1][1].append(old[-1])
                    # if service:
                    samples[-1][0].append(dialogue["utterance"])
                    samples[-1][1].append(service)
                else:
                    # print("service", service)
                    samples[-1][0].append(dialogue["utterance"])
                    samples[-1][1].append(service)
            samples[-1][0].append(interviewer(cur_spl))
            samples[-1][1].append(None)

new_samples = []
for sample in samples:
    tmp = [sample[0][0] + sample[0][1]]
    new_samples.append((tmp + sample[0][2:], sample[1][1:]))
with open("samples.pkl", "wb") as f:
    pickle.dump(new_samples, f)
    
grouped_samples = []
group_size = 10  
for i in range(0, len(new_samples), group_size):
    group = new_samples[i:i+group_size] 
    if group:  
        first_question = group[0][0][0] 
        reminder_question = f"Do you remember the answer to the first question: '{first_question}'?"
        group.append(([reminder_question], [None]))
    grouped_samples.append(group)
with open("grouped_samples.pkl", "wb") as f:
    pickle.dump(grouped_samples, f)






