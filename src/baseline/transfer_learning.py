'''
Idea:
Start with inceptionnet/resnet/vgg pretrained on imagenet: https://keras.io/applications/
add own classifier on top of it: flatten, dense, relu, dropout, dense, sigmoid
train own classifier, freeze feature extractor net
train own classifier and last convnet block
(train with small learning rate and sgd to not destroy previously learned features)

Be happy and hopefully win the competition ;)
'''

import pandas as pd
import os
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential

base_datasets_dir = '/var/tmp/studi5/boneage/datasets/'

base_chest_dir = base_datasets_dir + 'nih-chest-xrays/'
chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'))
print(chest_df.as_matrix(['Image Index'])[0:3])
print(chest_df.as_matrix(['Patient Age'])[0:3])
print(chest_df.as_matrix(['Image Index', 'Patient Age'])[0:3])

raw_train_df, valid_df = train_test_split(chest_df, test_size=0.2, random_state=2018)  # , stratify=chest_df['Patient Age'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])






base_boneage_dir = base_datasets_dir + 'boneage/'
boneage_df = pd.read_csv(os.path.join(base_boneage_dir, 'boneage-training-dataset.csv')) # read csv
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_boneage_dir, 'boneage-training-dataset',
                                                         '{}.png'.format(x))) # add path to dictionary
boneage_df['exists'] = boneage_df['path'].map(os.path.exists) # add exists to dictionary
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total') # print how many images have been found
boneage_df['gender'] = boneage_df['male'].map(lambda x: 'male' if x else 'female') # convert boolean to string male or female
boneage_df['boneage_category'] = pd.cut(boneage_df['boneage'], 10)





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