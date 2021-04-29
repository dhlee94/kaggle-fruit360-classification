import pandas as pd
import os
from glob import glob

DATA_PATH = "./Data/Test"
data_list = os.listdir(DATA_PATH)

file_list_dict = {"image": [], "label": []}
for data in data_list:
    file_list = glob(DATA_PATH+"/"+data+"/"+"*.jpg")
    for file in file_list:
        file_list_dict["image"].append(file)
        file_list_dict["label"].append(data_list.index(data))

train_data = pd.DataFrame(data=file_list_dict)
train_data.to_csv("./test_data.csv", index=False)


