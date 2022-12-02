import json

path = "train_data_all.json"
with open(path, 'r') as f:
    dataset = json.load(f)
    print(dataset)
