'''
Idea:
Start with inceptionnet/resnet/vgg pretrained on imagenet: https://keras.io/applications/
add own classifier on top of it: flatten, dense, relu, dropout, dense, sigmoid
train own classifier, freeze feature extractor net
train own classifier and last convnet block
(train with small learning rate and sgd to not destroy previously learned features)

Inspiration: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Be happy and hopefully win the competition ;)
'''

import pandas as pd
import os
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential

base_datasets_dir = '/var/tmp/studi5/boneage/datasets/'

''' ==================================================
Read RSNA Chest XRays Dataset
================================================== '''
base_chest_dir = base_datasets_dir + 'nih-chest-xrays/'

chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'))
chest_df['Patient Age'] = [int(x[:-1]) * 12 for x in chest_df['Patient Age']]  # parse Year Patient Age to Month age
chest_df['path'] = chest_df['Image Index'].map(
    lambda x: os.path.join(base_chest_dir, 'images', x))  # create path from id
chest_df['exists'] = chest_df['path'].map(os.path.exists)
print(chest_df['exists'].sum(), 'images found of', chest_df.shape[0], 'total')
# chest_df['chest_category'] = pd.cut(chest_df['Patient Age'], 10)

raw_train_df, valid_df = train_test_split(chest_df, test_size=0.2,
                                          random_state=2018)  # , stratify=chest_df['chest_category'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

''' ==================================================
Read RSNA Boneage Dataset
================================================== '''
base_boneage_dir = base_datasets_dir + 'boneage/'

boneage_df = pd.read_csv(os.path.join(base_boneage_dir, 'boneage-training-dataset.csv'))
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_boneage_dir, 'boneage-training-dataset',
                                                                 '{}.png'.format(x)))  # create path from id
boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
# boneage_df['boneage_category'] = pd.cut(boneage_df['boneage'], 10)

raw_train_df, valid_df = train_test_split(boneage_df, test_size=0.2,
                                          random_state=2018)  # ,stratify=boneage_df['boneage_category'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

''' ==================================================
Preprocess Image Data
================================================== '''
IMG_SIZE = (384, 384)  # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False,
                              samplewise_std_normalization=False,
                              horizontal_flip=True,
                              vertical_flip=False,
                              height_shift_range=0.15,
                              width_shift_range=0.15,
                              rotation_range=5,
                              shear_range=0.01,
                              fill_mode='nearest',
                              zoom_range=0.25,
                              preprocessing_function=preprocess_input)

''' ==================================================
Build Model
================================================== '''
model = Sequential()
conv_base_model = InceptionResNetV2(include_top=True,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=1000)
conv_base_model.trainable = False

model.add(conv_base_model)
model.add(Dense(1, kernel_initializer='normal'))
