from glob import glob
import os
import pandas as pd

path = './Data/Training'
dataset_list = glob(os.path.join(path, "*"))
class_list = []
class_num = 0

dataset = {'images': [], 'label': []}
for i in dataset_list:
    data_list = glob(os.path.join(i, "*.jpg"))
    for i in data_list:
        dataset['images'].append(i)
        dataset['label'].append(class_num)
    class_num += 1

df = pd.DataFrame(dataset)
df.to_csv("./training.csv", index=False)