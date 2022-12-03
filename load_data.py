import json

path = "train_data_all.json"
with open(path, 'r') as f:
    dataset = json.load(f)
    # print(dataset)
# print(len(dataset))

count = 0
for val in dataset:
    if val['weight'] != '' and val['height'] != '' and val['fit'] != '' and val['size'] != '' and val['size'] != 'NONE' and val['usually_wear'] != '':
        count += 1

print(count)

