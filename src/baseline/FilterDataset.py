# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # showing and rendering figures

# io related
from skimage.io import imread
import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input
##split data into training and validation
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, \
    Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.metrics import mean_absolute_error

import pickle
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_hist, equalize_adapthist
from keras.models import load_model

import ImageSelector as imgsel

chunksize = 20


dir_path = os.getcwd()

if dir_path[1]=='h':
    server = False
    base_bone_dir = '/home/guy/jmcs-atml-bone-age-prediction/datasets/'
else:
    server = True
    base_bone_dir = '/var/tmp/studi5/boneage/datasets/boneage/'
    
model_prediction_dir = 'ModelPrediction/'
    
model_accuracy = load_model(base_bone_dir + model_prediction_dir+ 'weights-03-0.55.h5')

file = os.path.join(base_bone_dir, 'boneage-training-dataset.csv')

iter_csv = pd.read_csv(file, iterator=True, chunksize=chunksize)

output_file = 'boneage-training-dataset-filtered.csv'

#output_dataframe = 
index = 0
for i in iter_csv:
    print('iteration' + str(index))
    dictio = i.to_dict()
    #for key in dict['id']:
    #    print('key')
    boneage = dictio['boneage']
    identity = dictio['id']
    male = dictio['male']
    
    list_images = [v for v in identity.values()]
    
    imgs = imgsel.LoadImg2Mem(list_images,384)
    
    accuracy_output = model_accuracy.predict(imgs)
    
    accuracy = accuracy_output[:,0]-accuracy_output[:,1]
    
    todel = []
    for key, item in identity.items():
        if key < chunksize:
            if accuracy[key]<0:
                todel.append(key)
    
    for i in todel:
        del boneage[i]
        del identity[i]
        del male[i]
    
    
    filtered_dico = {'boneage': boneage, 'id': identity, 'male': male}
    
    
    # Convert to dataframe
    new_dataframe = pd.DataFrame.from_dict(filtered_dico)
    if index ==0:
        output_dataframe = new_dataframe
    else:
        output_dataframe = pd.concat([output_dataframe, new_dataframe])
    #df = pd.concat([chunk[chunk[0] >  10] for chunk in iter_csv])
    index += 1
    
output_dataframe.to_csv(base_bone_dir+output_file, sep='\t')

print('EOF')