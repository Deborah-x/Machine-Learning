import json

def load_dataset():
    path = "train_data_all.json"
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset

def key_num(*args):
    print(args)
    count = 0
    for index in load_dataset():
        flag = 1
        for key in args:
            if index[key] == '':
                flag = 0
                break
        count += flag
    return count

print(key_num('fit', 'weight', 'height', 'size', 'usually_wear'))
