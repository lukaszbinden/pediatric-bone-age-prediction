from itertools import islice, chain

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from itertools import cycle

from keras import Input
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D, concatenate
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.metrics import mean_absolute_error
from transfer_learning_common import flow_from_dataframe


tstart = datetime.now()
# hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
base_dir = '/var/tmp/studi5/boneage/'
base_datasets_dir = base_dir + '/datasets/'
chest_dataset_dir = base_datasets_dir + 'nih-chest-xrays/'  # 'nih-chest-xrays-full/'

# default size of InceptionResNetV2
# cf. https://stackoverflow.com/questions/43922308/what-input-image-size-is-correct-for-the-version-of-resnet-v2-in-tensorflow-slim
IMG_SIZE = (299, 299)


class_str_col_boneage = 'boneage'
class_str_col_chest = 'Patient Age'
disease_str_col = 'Finding Labels'
gender_str_col_chest = 'Patient Gender'
gender_str_col = gender_str_col_chest


def get_chest_dataframe():
    img_dir = 'images'
    csv_name = 'sample_labels_sm.csv'
    image_index_col = 'Image Index'

    diseases = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

    chest_df = pd.read_csv(os.path.join(chest_dataset_dir, csv_name),
                           usecols=[image_index_col, class_str_col_chest, gender_str_col_chest, disease_str_col])
    chest_df[class_str_col_chest] = [int(x[:-1] if type(x) == str and x[-1] == 'Y' else x) * 12 for x in
                                     chest_df[class_str_col_chest]]  # parse Year Patient Age to Month age

    chest_df['path'] = chest_df[image_index_col].map(
        lambda x: os.path.join(chest_dataset_dir, img_dir, x))  # create path from id
    chest_df['exists'] = chest_df['path'].map(os.path.exists)
    print('chest', chest_df['exists'].sum(), 'images found of', chest_df.shape[0], 'total')
    chest_df[gender_str_col_chest] = chest_df[gender_str_col_chest].map(
        lambda x: np.array([1]) if x == 'M' else np.array([0]))  # map 'M' and 'F' values to 1 and 0

    #print(chest_df[disease_str_col])

    chest_df[disease_str_col] = [np.array([1 if disease in x else 0 for disease in diseases]) for x in chest_df[
        disease_str_col]]  # convert diseases string into sparse binary vector for classification

    #chest_df[disease_str_col] = chest_df.drop(chest_df[chest_df[disease_str_col] == np.array(
    #    [0] * 14)].index)  # delete all rows with zero entries (no disease)

    return chest_df


core_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                              zoom_range=0.2, horizontal_flip=True)

val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)


df = get_chest_dataframe()
class_str_col = disease_str_col
train_df, val_df = train_test_split(df, test_size=0.2, random_state=2018)

train_gen = flow_from_dataframe(core_idg, train_df, path_col='path', y_col=class_str_col,
                                target_size=IMG_SIZE,
                                color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

val_gen = flow_from_dataframe(val_idg, val_df, path_col='path', y_col=class_str_col,
                              target_size=IMG_SIZE,
                              color_mode='rgb',
                              batch_size=BATCH_SIZE_VAL)

input_gender = Input(shape=(1,), name='input_gender')
i1 = Input(shape=(299, 299, 3), name='input_img')
inputs = [i1, input_gender]
base = InceptionV3(input_tensor=i1, input_shape=(299, 299, 3), include_top=False, weights=None)

num_trainable_layers = 5
base.trainable = True
for layer in base.layers[0:len(base.layers) - num_trainable_layers]:
    layer.trainable = False
for layer in base.layers[-num_trainable_layers:]:
    layer.trainable = True

feature_img = base.get_layer(name='mixed10').output
feature_img = AveragePooling2D((2, 2))(feature_img)
feature_img = Flatten()(feature_img)
feature_gender = Dense(32, activation='relu')(input_gender)
feature = concatenate([feature_img, feature_gender], axis=1)
# feature = feature_img
o = Dense(1000, activation='relu')(feature)
o = Dense(1000, activation='relu')(o)
o = Dense(14, name='output_disease', activation='sigmoid')(o)
model = Model(inputs=inputs, outputs=o)
# model = Model(inputs=i1, outputs=o)
# optimizer = Adam(lr=1e-3)
# model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae'])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()  # prints the network structure

earlyStopping = EarlyStopping(monitor="val_loss", mode="min",
                              patience=10)

reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
                                      epsilon=0.0001,
                                      cooldown=5, min_lr=LEARNING_RATE * 0.1)


def combined_generators(image_generator, gender, disease, batch_size):
    gender_generator = cycle(batch(gender, batch_size))
    disease_generator = cycle(batch(disease, batch_size))
    while True:
        nextImage = next(image_generator)
        # print('nextImage[0]')
        # print(nextImage[0])
        # print('nextImage[1]')
        # print(nextImage[1])
        nextGender = next(gender_generator)
        # print('nextGender')
        # print(nextGender)
        # print('np.stack(nextGender.values)')
        # print(np.stack(nextGender.values))
        assert len(nextImage[0]) == len(nextGender)
        nextDisease = next(disease_generator)
        # print('nextDisease')
        # print(nextDisease)
        # print('np.stack(nextDisease.values)')
        # print(np.stack(nextDisease.values))
        assert len(nextImage[0]) == len(nextDisease)
        yield [nextImage[0], np.stack(nextGender.values)], np.stack(nextDisease.values)
        # yield nextImage[0], np.stack(nextDisease.values)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# print('train_gen.classes[:10]')
# print(train_gen.classes[:10])
# print('train_df[disease_str_col].values')
# print(train_df[disease_str_col].values)
# print('np.stack(train_df[disease_str_col].values)')
# print(np.stack(train_df[disease_str_col].values))
# print('train_df[disease_str_col].values')
# print(train_df[disease_str_col].values)

train_gen_wrapper = combined_generators(train_gen, train_df[gender_str_col], train_df[disease_str_col], BATCH_SIZE_TRAIN)
val_gen_wrapper = combined_generators(val_gen, val_df[gender_str_col], val_df[disease_str_col], BATCH_SIZE_VAL)
# train_gen_wrapper = train_gen
# val_gen_wrapper = val_gen

history = model.fit_generator(train_gen_wrapper, validation_data=val_gen_wrapper,
                              epochs=NUM_EPOCHS, verbose=1,
                              steps_per_epoch=len(train_gen),
                              validation_steps=len(val_gen),
                              callbacks=[earlyStopping, reduceLROnPlateau])  # trains the model


print('result: ', history.history['val_acc'][-1])
