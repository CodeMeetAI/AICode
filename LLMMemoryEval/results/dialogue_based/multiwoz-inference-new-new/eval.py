import os
import json

for file in sorted(os.listdir(".")):
    if ".jsonl" in file:
        accuracy = []
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                accuracy.append(data['gt'] == data['answer'])

        print(file, sum(accuracy)/len(accuracy))
