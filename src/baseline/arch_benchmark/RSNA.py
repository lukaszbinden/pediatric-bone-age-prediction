import numpy as np
import csv
import random
from PIL import Image
import scipy
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import Input, Dense, Flatten, Activation,\
    BatchNormalization, Reshape, UpSampling2D, ZeroPadding2D, \
    Dropout, Lambda, AveragePooling2D, GlobalAveragePooling2D, concatenate
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, CSVLogger, RemoteMonitor,ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
from se_resnet_rsna import SEResNet50
from ResnetXtrsna import ResNextImageNet
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# config = tf.ConfigProto ()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# sess = tf.Session (config=config)
# KTF.set_session (sess)

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
def LoadData2Mem(data_list_use, img_size=500):
    img_data=[]
    boneage_data=[]
    gender_data=[]
    for i in range(len(data_list_use)):
        img_name = base_bone_dir+'/boneage-training-dataset/'+data_list_use[i][0]+'.png'
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
    img_data = (np.array(np.reshape(img_data, (-1, 500, 500, 1)), dtype='float32')/255.0)-0.5
    boneage_data = np.array(np.reshape(boneage_data,(-1,)), dtype='float32')
    gender_data = np.array(np.reshape(gender_data,(-1,)), dtype='float32')
    return img_data, boneage_data, gender_data


def SaveDataList(path, data_list):
    csvFile = open (path, 'w')
    writer = csv.writer (csvFile)
    writer.writerows (data_list)
    csvFile.close()
def boneage_seResNet_model():
    i1 = Input (shape=(500, 500, 1), name='input_img')
    base = SEResNet50(include_top=False, weights=None, input_tensor=i1, input_shape=(500, 500, 1))
    feature_img = base.output
    feature_img = AveragePooling2D ((4, 4)) (feature_img)
    feature_img = Flatten () (feature_img)
    feature = feature_img
    o = Dense (1000, activation='relu') (feature)
    o = Dense (1000, activation='relu') (o)
    o = Dense (1) (o)

    model = Model (inputs=i1, outputs=o)
    optimizer = Adam (lr=0.00005)
    #model = multi_gpu_model(model, gpus=2)
    model.compile (loss='mean_absolute_error', optimizer=optimizer)
    return model
def boneage_ResNetxt_model():

    i1 = Input (shape=(500, 500, 1), name='input_img')
    base = ResNextImageNet(include_top=False, weights=None, input_tensor=i1, input_shape=(500, 500, 1))
    feature_img = base.output
    feature_img = AveragePooling2D ((4, 4)) (feature_img)
    feature_img = Flatten () (feature_img)
    feature = feature_img
    o = Dense (1000, activation='relu') (feature)
    o = Dense (1000, activation='relu') (o)
    o = Dense (1) (o)
    model = Model (inputs=i1, outputs=o)
    optimizer = Adam ()
    model.compile (loss='mean_absolute_error', optimizer=optimizer)
    return model
def Boneage_ResNet_model():
    i1 = Input (shape=(500, 500, 1), name='input_img')
    base = ResNet50(include_top=False,weights=None, input_tensor=i1, input_shape=(500, 500, 1))
    feature_img = base.get_layer(name='activation_49').output
    feature_img = AveragePooling2D ((4, 4)) (feature_img)
    feature_img = Flatten () (feature_img)
    feature = feature_img
    o = Dense (1000, activation='relu') (feature)
    o = Dense (1000, activation='relu') (o)
    o = Dense (1) (o)
    model = Model (inputs=i1, outputs=o)
    optimizer = Adam ()
    model.compile (loss='mean_absolute_error', optimizer=optimizer)
    return model
def Boneage_prediction_model():
    i1 = Input(shape=(500,500,1), name='input_img')
    #i2 = Input(shape=(1,), name='input_gender')
    base = InceptionV3(input_tensor=i1, input_shape=(500,500,1), include_top=False, weights=None)
    feature_img = base.get_layer(name='mixed10').output
    # feature_img = AveragePooling2D((2,2), name='ave_pool_fea')(feature_img)
    # feature_img = Flatten()(feature_img)
    #feature_img = GlobalAveragePooling2D()(feature_img)
    feature_img = AveragePooling2D((2,2))(feature_img)
    feature_img = Flatten()(feature_img)
    #feature_gender = Dense(32, activation='relu')(i2)
    #feature = concatenate([feature_img, feature_gender], axis=1)
    feature = feature_img
    o = Dense(1000, activation='relu')(feature)
    o = Dense(1000, activation='relu')(o)
    o = Dense(1)(o)
    model = Model(inputs=i1, outputs=o)
    optimizer =Adam(lr=1e-3)
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model

train_list_use = LoadDataList('boneage_train_list_use.csv')
val_list_use = LoadDataList('boneage_val_list_use.csv')

num_train_sample = len(train_list_use)
num_val_sample = len(val_list_use)
command='test'
if command=='train':
    #model = boneage_ResNetxt_model ()
    model = boneage_seResNet_model()
    model.load_weights ('/home/luya/food-recognition-madima2016/boneage/weights_resse50-22-13.05.h5')
    img_train, boneage_train, gender_train=LoadData2Mem(train_list_use, 500)
    img_val, boneage_val, gender_val=LoadData2Mem(val_list_use, 500)

    #model = Boneage_prediction_model()
    #model = Boneage_ResNet_model()

    history = History()
    logger = CSVLogger(base_bone_dir+'/log_resse50GN.csv', separator=',', append=False)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=10, mode='min')
    checkpoint = ModelCheckpoint(base_bone_dir+'weights_resse50GN-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min',save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

    datagen = ImageDataGenerator(  # 1
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(img_train)
    #
    model.fit_generator(datagen.flow(img_train, boneage_train, batch_size=16),
                        steps_per_epoch = 600,
                        epochs=50,
                        callbacks=[history, logger, earlystopping, checkpoint, reduceLROnPlat],
                        validation_data=(img_val, boneage_val),
                        verbose=1)

else:
    model = Boneage_prediction_model()
    model.summary ()