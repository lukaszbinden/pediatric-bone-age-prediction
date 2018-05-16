#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 09:14:17 2018

@author: guy
"""

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras import backend as K

import csv
from PIL import Image
import numpy as np
import scipy

import pickle

# obj0, obj1, obj2 are created here...

batch_size = 128
num_classes = 1
epochs = 12

base_bone_dir = '/home/guy/jmcs-atml-bone-age-prediction/datasets/'
path_var_store = '/home/guy/jmcs-atml-bone-age-prediction/variables/'
# input image dimensions
img_rows, img_cols = 384, 384
img_size_bone_age_model = 384

def convert_gray_to_rgb(im_list):
    # I think this will be slow
    dim1, dim2, dim3, dim4 = im_list.shape
    imglistrgb = np.zeros((dim1,dim2,dim3,3))
    
    for i,j in enumerate(imglistrgb):
        im = im_list[i,:,:]
        im2 = np.append(im,im,axis=2)
        im3 = np.append(im2,im,axis=2)
        imglistrgb[i] = im3
        
    return imglistrgb

def Ygenerator(model, boneage_train, img_train , tolerance):
    
    #img_list = to_rgb1(im_train)
    img_train = convert_gray_to_rgb(img_train)
    Y_pred = model.predict(img_train)
    Y = []
    for i in range(len(Y_pred)):
        if Y_pred[i][0] < boneage_train[i]*(1+tolerance) and Y_pred[i][0] > boneage_train[i]*(1-tolerance):
            #good estimation
            Y.append(True)
        else:
            #bad estimation
            Y.append(False)
    return Y

def TrainPredictorModel(img_train, errorpred_train, img_val, errorpred_val):
    # Train Dataset to recognize accuracy
    model_predictor = ImageSelectorModel()

    #history = History()
    #history = bone_age_model.fit_generator(train_gen, validation_data=(test_X, test_Y))
    logger = CSVLogger(base_bone_dir+'/log.csv', separator=',', append=False)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=10, mode='min')
    checkpoint = ModelCheckpoint(base_bone_dir+'weights-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

    model_predictor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #imgsel.TrainImageSelector(model_predictor, NextSet_X, NextSet_Y, NextSet_X, NextSet_Y)
    datagen = ImageDataGenerator(#featurewise_center = True/False
                              samplewise_center=False,
                              #featurewise_std_normalization = True,
                              samplewise_std_normalization=True,
                              #zca_epsilon=True,
                              #zca_whitening=True,
                              rotation_range=20,
                              width_shift_range=0.15,
                              #float = ,
                              #int = ,
                              #shear_range=0.01,
                              zoom_range=0.25,
                              #channel_shift-range = 
                              fill_mode='nearest',
                              #cval =   
                              horizontal_flip=True,
                              vertical_flip=False,
                              #rescale = 
                              #preprocessing_function=prepro,
                              #data_format = 
                              #validation_split
                              height_shift_range=0.15
                              )
    
    datagen.fit(img_train)
    
    model_predictor.fit_generator(datagen.flow(img_train, errorpred_train, batch_size=16),steps_per_epoch = 600,epochs=3,callbacks=[logger, earlystopping, checkpoint, reduceLROnPlat],validation_data=(img_val, errorpred_val),verbose=1)
    
    scores = model_predictor.evaluate(img_val, errorpred_val, verbose=1)

def ExtractimageQuality(model_bone_age):
    # ------------------------------------------
    # Test Model
    # ------------------------------------------
    tolerance = 0.3
    
    list_data = LoadDataList('boneage-training-dataset.csv')#need to be changed
    
    # Reduced train list
    train_list = dict((k, v) for k, v in list_data.items() if k > 0 and k < 300)
    val_list = dict((k, v) for k, v in list_data.items() if k > 300 and k < 350)
    #if epoch == 0:
    
    img_train, boneage_train, gender_train= LoadData2Mem(train_list, img_size_bone_age_model)
    img_val, boneage_val, gender_val= LoadData2Mem(val_list, img_size_bone_age_model)
    
    errorpred_train = Ygenerator(model_bone_age, boneage_train, img_train, tolerance)
    errorpred_val = Ygenerator(model_bone_age,boneage_train, img_val, tolerance)

    # Saving the objects:
    with open(path_var_store + 'objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([img_train, errorpred_train, img_val, errorpred_val], f)

    TrainPredictorModel(img_train, errorpred_train, img_val, errorpred_val)
    
    
    
  

def ImageSelectorModel():
    input_shape = (img_rows, img_cols,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def TrainImageSelector(model, xtrain, ytrain, xtest, ytest):
    x_train = xtrain
    y_train = ytrain
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest, ytest))
    

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

def LoadData2Mem(data_list_use, img_size=500):
    img_data=[]
    boneage_data=[]
    gender_data=[]
    for i in data_list_use:
        img_name = base_bone_dir+'boneage-training-dataset/'+data_list_use[i][0]+'.png'
        img = Image.open(img_name).convert('L')
        img = np.array(scipy.misc.imresize (img, (img_size, img_size)))
        boneage =  int(data_list_use[i][1])
        gender = data_list_use[i][2]
        if gender=='True': ##is male
            gender_int=1
        else:
            gender_int=0
        img_data.append(img)
        boneage_data.append(boneage)
        gender_data.append(gender_int)
    img_data = (np.array(np.reshape(img_data, (-1, img_size_bone_age_model, img_size_bone_age_model, 1)), dtype='float32')/255.0)-0.5
    boneage_data = np.array(np.reshape(boneage_data,(-1,)), dtype='float32')
    gender_data = np.array(np.reshape(gender_data,(-1,)), dtype='float32')
    return img_data, boneage_data, gender_data


#DEBUGGING
# Getting back the objects:

print("---------------------------------------")
print("----------------Debug------------------")
print("---------------------------------------")
'''
with open(path_var_store + 'objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    img_train, errorpred_train, img_val, errorpred_val = pickle.load(f)

TrainPredictorModel(img_train, errorpred_train, img_val, errorpred_val)
'''