import numpy as np
import pandas as pd
import os
from datetime import datetime

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
from transfer_learning_common import flow_from_dataframe, get_chest_dataframe

tstart = datetime.now()

# hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
base_dir = '/var/tmp/studi5/boneage/'
base_datasets_dir = base_dir + '/datasets/'

# default size of InceptionResNetV2
# cf. https://stackoverflow.com/questions/43922308/what-input-image-size-is-correct-for-the-version-of-resnet-v2-in-tensorflow-slim
IMG_SIZE = (299, 299)

print('==================================================')
print('============ Preprocessing Image Data ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).

# core_idg = ImageDataGenerator(samplewise_center=False,
#                               samplewise_std_normalization=False,
#                               horizontal_flip=True,
#                               vertical_flip=False,
#                               height_shift_range=0.15,
#                               width_shift_range=0.15,
#                               rotation_range=5,
#                               shear_range=0.01,
#                               fill_mode='nearest',
#                               zoom_range=0.25,
#                               preprocessing_function=preprocess_input)

core_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                              zoom_range=0.2, horizontal_flip=True)

val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

print('==================================================')
print('============ Creating Data Generators ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

print('==================================================')
print('========== Reading RSNA Boneage Dataset ==========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

base_boneage_dir = base_datasets_dir + 'boneage/'
class_str_col = 'boneage'
gender_str_col = 'gender'

boneage_df = pd.read_csv(os.path.join(base_boneage_dir, 'boneage-training-dataset.csv'))
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_boneage_dir, 'boneage-training-dataset',
                                                                 '{}.png'.format(x)))  # create path from id

boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
# boneage_df['boneage_category'] = pd.cut(boneage_df[class_str_col], 10)

train_df_boneage, valid_df_boneage = train_test_split(boneage_df, test_size=0.2,
                                                      random_state=2018)  # ,stratify=boneage_df['boneage_category'])
print('train', train_df_boneage.shape[0], 'validation', valid_df_boneage.shape[0])

train_gen_boneage = flow_from_dataframe(core_idg, train_df_boneage, path_col='path', y_col=class_str_col, gender_col = gender_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

# used a fixed dataset for evaluating the algorithm
valid_gen_boneage = flow_from_dataframe(core_idg, valid_df_boneage, path_col='path', y_col=class_str_col, gender_col = gender_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb',
                                        batch_size=BATCH_SIZE_VAL)  # we can use much larger batches for evaluation

print('==================================================')
print('================= Building Model =================')
print('==================================================')

print('current time: %s' % str(datetime.now()))

print(next(train_gen_boneage))

i1 = Input(shape=(299, 299, 3), name='input_img')
i2 = Input(shape=(1,), name='input_gender')
base = InceptionV3(input_tensor=i1, input_shape=(299, 299, 3), include_top=False, weights=None)
feature_img = base.get_layer(name='mixed10').output
# feature_img = AveragePooling2D((2,2), name='ave_pool_fea')(feature_img)
# feature_img = Flatten()(feature_img)
# feature_img = GlobalAveragePooling2D()(feature_img)
feature_img = AveragePooling2D((2, 2))(feature_img)
feature_img = Flatten()(feature_img)
feature_gender = Dense(32, activation='relu')(i2)
feature = concatenate([feature_img, feature_gender], axis=1)
#feature = feature_img
o = Dense(1000, activation='relu')(feature)
o = Dense(1000, activation='relu')(o)
o = Dense(1)(o)
model = Model(inputs=i1, outputs=o)
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_absolute_error', optimizer=optimizer)

print('==================================================')
print('======= Training Model on Boneage Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

model.summary()

weight_path = base_dir + "{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

early = EarlyStopping(monitor="val_loss", mode="min",
                      patience=5)  # probably needs to be more patient, but kaggle time is limited

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1,
                                   mode='auto', epsilon=0.0001, cooldown=5, min_lr=LEARNING_RATE * 0.1)

history = model.fit_generator(train_gen_boneage, validation_data=valid_gen_boneage, epochs=NUM_EPOCHS,
                              callbacks=[checkpoint, early, reduceLROnPlat])
print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])

print('==================================================')
print('================ Evaluating Model ================')
print('==================================================')

tend = datetime.now()
print('current time: %s' % str(datetime.now()))
print('elapsed time: %s' % str((tend - tstart)))
