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

import numpy as np
import pandas as pd
import os
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential

base_datasets_dir = '/var/tmp/studi5/boneage/datasets/'
IMG_SIZE = (384, 384)  # slightly smaller than vgg16 normally expects

print('==================================================')
print('============ Preprocessing Image Data ============')
print('==================================================')
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

print('==================================================')
print('============ Creating Data Generators ============')
print('==================================================')


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = base_dir  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


print('==================================================')
print('======== Reading NIH Chest XRays Dataset =========')
print('==================================================')
base_chest_dir = base_datasets_dir + 'nih-chest-xrays/'
class_str = 'Patient Age'

chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'))
chest_df[class_str] = [int(x[:-1]) * 12 for x in chest_df[class_str]]  # parse Year Patient Age to Month age
chest_df['path'] = chest_df['Image Index'].map(
    lambda x: os.path.join(base_chest_dir, 'images', x))  # create path from id
chest_df['exists'] = chest_df['path'].map(os.path.exists)
print(chest_df['exists'].sum(), 'images found of', chest_df.shape[0], 'total')
# chest_df['chest_category'] = pd.cut(chest_df[class_str], 10)

train_df_chest, valid_df_chest = train_test_split(chest_df, test_size=0.2,
                                                  random_state=2018)  # , stratify=chest_df['chest_category'])
print('train', train_df_chest.shape[0], 'validation', valid_df_chest.shape[0])

train_gen_chest = flow_from_dataframe(core_idg, train_df_chest, path_col='path', y_col=class_str, target_size=IMG_SIZE,
                                      color_mode='rgb', batch_size=32)

valid_gen_chest = flow_from_dataframe(core_idg, valid_df_chest, path_col='path', y_col=class_str, target_size=IMG_SIZE,
                                      color_mode='rgb', batch_size=256)  # we can use much larger batches for evaluation

print('==================================================')
print('========== Reading RSNA Boneage Dataset ==========')
print('==================================================')
base_boneage_dir = base_datasets_dir + 'boneage/'
class_str = 'boneage'

boneage_df = pd.read_csv(os.path.join(base_boneage_dir, 'boneage-training-dataset.csv'))
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_boneage_dir, 'boneage-training-dataset',
                                                                 '{}.png'.format(x)))  # create path from id

boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
# boneage_df['boneage_category'] = pd.cut(boneage_df[class_str], 10)

train_df_boneage, valid_df_boneage = train_test_split(boneage_df, test_size=0.2,
                                                      random_state=2018)  # ,stratify=boneage_df['boneage_category'])
print('train', train_df_boneage.shape[0], 'validation', valid_df_boneage.shape[0])

train_gen_boneage = flow_from_dataframe(core_idg, train_df_boneage, path_col='path', y_col=class_str,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=32)

valid_gen_boneage = flow_from_dataframe(core_idg, valid_df_boneage, path_col='path', y_col=class_str,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb',
                                        batch_size=256)  # we can use much larger batches for evaluation

print('==================================================')
print('================= Building Model =================')
print('==================================================')
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

print('==================================================')
print('================= Training Model =================')
print('==================================================')

print('==================================================')
print('================ Evaluating Model ================')
print('==================================================')
