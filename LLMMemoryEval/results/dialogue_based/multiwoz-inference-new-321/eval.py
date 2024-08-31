import json
import os

for file in sorted(os.listdir(".")):
    accuracy = []
    if "jsonl" in file:
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                accuracy.append(data['answer'] == data['gt'])

        print(file, sum(accuracy)/len(accuracy))
