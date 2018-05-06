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
from datetime import datetime

from keras import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.metrics import mean_absolute_error

tstart = datetime.now()

# hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 256
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

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    """
    Creates a DirectoryIterator from in_df at path_col with image preprocessing defined by img_data_gen. The labels
    are specified by y_col.

    :param img_data_gen: an ImageDataGenerator
    :param in_df: a DataFrame with images
    :param path_col: name of column in in_df for path
    :param y_col: name of column in in_df for y values/labels
    :param dflow_args: additional arguments to flow_from_directory
    :return: df_gen (keras.preprocessing.image.DirectoryIterator)
    """
    print('flow_from_dataframe() -->')
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    # flow_from_directory: Takes the path to a directory, and generates batches of augmented/normalized data.
    # sparse: a 1D integer label array is returned
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    # df_gen: A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images
    # with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = base_dir  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    print('flow_from_dataframe() <--')
    return df_gen


print('==================================================')
print('======== Reading NIH Chest XRays Dataset =========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

base_chest_dir = base_datasets_dir + 'nih-chest-xrays/'
image_index_col = 'Image Index'
class_str_col = 'Patient Age'

chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'), usecols=[image_index_col, class_str_col])
chest_df[class_str_col] = [int(x[:-1]) * 12 for x in chest_df[class_str_col]]  # parse Year Patient Age to Month age
chest_df['path'] = chest_df[image_index_col].map(
    lambda x: os.path.join(base_chest_dir, 'images', x))  # create path from id
chest_df['exists'] = chest_df['path'].map(os.path.exists)
print(chest_df['exists'].sum(), 'images found of', chest_df.shape[0], 'total')
# chest_df['chest_category'] = pd.cut(chest_df[class_str], 10)

raw_train_df_chest, valid_df_chest = train_test_split(chest_df, test_size=0.2,
                                                      random_state=2018)  # , stratify=chest_df['chest_category'])
print('train_chest', raw_train_df_chest.shape[0], 'validation_chest', valid_df_chest.shape[0])

# NO Balance the distribution in the training set AT THIS POINT
train_df_chest = raw_train_df_chest

train_gen_chest = flow_from_dataframe(core_idg, train_df_chest, path_col='path', y_col=class_str_col,
                                      target_size=IMG_SIZE,
                                      color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)


valid_gen_chest = flow_from_dataframe(val_idg, valid_df_chest, path_col='path', y_col=class_str_col,
                                      target_size=IMG_SIZE,
                                      color_mode='rgb',
                                      batch_size=BATCH_SIZE_VAL)  # we can use much larger batches for evaluation

print('==================================================')
print('========== Reading RSNA Boneage Dataset ==========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

base_boneage_dir = base_datasets_dir + 'boneage/'
class_str_col = 'boneage'

boneage_df = pd.read_csv(os.path.join(base_boneage_dir, 'boneage-training-dataset.csv'))
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_boneage_dir, 'boneage-training-dataset',
                                                                 '{}.png'.format(x)))  # create path from id

boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
# boneage_df['boneage_category'] = pd.cut(boneage_df[class_str_col], 10)

train_df_boneage, valid_df_boneage = train_test_split(boneage_df, test_size=0.2,
                                                      random_state=2018)  # ,stratify=boneage_df['boneage_category'])
print('train', train_df_boneage.shape[0], 'validation', valid_df_boneage.shape[0])

train_gen_boneage = flow_from_dataframe(core_idg, train_df_boneage, path_col='path', y_col=class_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

# used a fixed dataset for evaluating the algorithm
valid_gen_boneage = flow_from_dataframe(core_idg, valid_df_boneage, path_col='path', y_col=class_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb',
                                        batch_size=BATCH_SIZE_VAL)  # we can use much larger batches for evaluation

print('==================================================')
print('================= Building Model =================')
print('==================================================')

print('current time: %s' % str(datetime.now()))

# t_x: ndarray of images
# t_y: ndarray of labels (patient age)
t_x, t_y = next(train_gen_chest)  # gets the next batch from the data generator
# t_x has 'channels_last' data format (default of InceptionResNetV2)
# print(t_x.shape[1:])
in_layer = Input(t_x.shape[1:])  # instantiate a Keras tensor

conv_base_model = InceptionResNetV2(include_top=True,
                                    # use default InceptionResNetV2 img size -- otherwise we would not be able to define our own input size!
                                    weights='imagenet',
                                    input_tensor=None,
                                    # input_shape=t_x.shape[1:],
                                    # pooling=None,
                                    # classes=1000
                                    )
conv_base_model.trainable = False

features = conv_base_model(in_layer)

classifier = Sequential()
print(conv_base_model.output_shape[1:])
classifier.add(Dense(256, activation='relu', input_shape=conv_base_model.output_shape[1:]))
# classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='relu'))

out_layer = classifier(features)

model = Model(inputs=[in_layer], outputs=[out_layer])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()  # prints the network structure

print('==================================================')
print('========= Training Model on Chest Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

weight_path = base_dir + "{}_weights.best.hdf5".format('chest_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                             save_weights_only=True)  # save the weights

early = EarlyStopping(monitor="val_loss", mode="min",
                      patience=5)  # probably needs to be more patient, but kaggle time is limited

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001,
                                   cooldown=5, min_lr=LEARNING_RATE)

history = model.fit_generator(train_gen_chest, validation_data=valid_gen_chest, epochs=NUM_EPOCHS,
                              callbacks=[checkpoint, early, reduceLROnPlat])  # trains the model
print('Chest dataset (fixed): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])
# ['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error', 'lr'


# make last couple of conv layers in resnet trainable -->
conv_base_model.trainable = True
for layer in conv_base_model.layers[0:len(conv_base_model.layers) - 5]:
    layer.trainable = False
for layer in conv_base_model.layers[-5:]:
    layer.trainable = True
# make last couple of conv layers in resnet trainable <--

model.compile(loss='mse', optimizer=SGD(lr=LEARNING_RATE * 0.1, momentum=0.9), metrics=['mae'])

model.summary()  # prints the network structure

print('re-train model -->')
history = model.fit_generator(train_gen_chest, validation_data=valid_gen_chest, epochs=NUM_EPOCHS,
                              callbacks=[checkpoint, early, reduceLROnPlat])  # trains the model
print('Chest dataset (finetuning): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])
print('re-train model <--')

print('==================================================')
print('======= Training Model on Boneage Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

<<<<<<< Updated upstream
model.compile(optimizer=Adam(lr=LEARNING_RATE*0.1), loss='mse', metrics=["mae"])
=======
adam = Adam(lr=LEARNING_RATE * 0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse', metrics=["mae"])
>>>>>>> Stashed changes

model.summary()

weight_path = base_dir + "{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

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
