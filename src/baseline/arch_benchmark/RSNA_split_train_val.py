import numpy as np
import csv
import random

base_bone_dir = '/home/luya/food-recognition-madima2016/boneage/'
def LoadDataList(path):
    train_csvFile = open (base_bone_dir + path, 'r')
    reader = csv.reader (train_csvFile)
    result = {}
    item_idx = 0
    for item in reader:
        if item_idx==0:
            item_idx+=1
            continue
        result[item_idx-1] = item
        item_idx += 1
    train_csvFile.close ()
    return result
def SaveDataList(path, data_list):
    csvFile = open (path, 'w')
    writer = csv.writer (csvFile)
    writer.writerows (data_list)
    csvFile.close()
train_list = LoadDataList('boneage-training-dataset.csv')

train_list_use=[]
val_list_use=[]
train_list_contain=np.zeros((len(train_list),),dtype='int32')
for i in range(len(train_list)):
    idx=random.randint(0, len(train_list)-1)
    train_list_use.append(train_list[idx])
    train_list_contain[idx]=1
for i in range(len(train_list)):
    if train_list_contain[i]==0:
        val_list_use.append(train_list[i])
SaveDataList(base_bone_dir + 'boneage_train_list_use.csv', train_list_use)
SaveDataList(base_bone_dir + 'boneage_val_list_use.csv', val_list_use)