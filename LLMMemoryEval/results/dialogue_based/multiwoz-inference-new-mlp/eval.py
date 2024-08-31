import os
import json

current_dir = os.path.dirname(__file__)

for file in sorted(os.listdir(current_dir)):
    if ".jsonl" in file:
        accuracy = []
        with open(os.path.join(current_dir, file), "r") as f:
            for line in f:
                data = json.loads(line)
                accuracy.append(data['gt'] == data['answer'])

        print(file, sum(accuracy)/len(accuracy))
