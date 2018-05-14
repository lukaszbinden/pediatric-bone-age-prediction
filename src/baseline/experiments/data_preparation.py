from itertools import cycle

import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split

base_dir = '/var/tmp/studi5/boneage/'
base_datasets_dir = base_dir + 'datasets/'
chest_dataset_dir = base_datasets_dir + 'nih-chest-xrays-full/'
boneage_dataset_dir = base_datasets_dir + 'boneage/'

class_str_col_boneage = 'boneage'
class_str_col_chest = 'Patient Age'

gender_str_col_boneage = 'male'
gender_str_col_chest = 'Patient Gender'

disease_str_col = 'Finding Labels'


def get_gen(train_idg, val_idg, img_size, batch_size_train, batch_size_val, dataset='boneage', disease_enabled=True):
    """
    :param train_idg:
    :param val_idg:
    :param img_size:
    :param batch_size_train:
    :param batch_size_val:
    :param dataset: either 'boneage' or 'chest' or 'chest_boneage_range'
    :param disease_enabled: True or False
    :return:
    """
    if dataset == 'boneage':
        df = get_boneage_dataframe()
        class_str_col = class_str_col_boneage
        gender_str_col = gender_str_col_boneage
    elif dataset == 'chest':
        df = get_chest_dataframe(False)
        class_str_col = class_str_col_chest
        gender_str_col = gender_str_col_chest
    elif dataset == 'chest_boneage_range':
        df = get_chest_dataframe(True)
        class_str_col = class_str_col_chest
        gender_str_col = gender_str_col_chest
    else:
        print('Please specify valid dataset name!')
        return

    y_cols = [class_str_col]
    if disease_enabled and dataset != 'boneage':
        y_cols.append(disease_str_col)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=2018)

    print('train', train_df.shape[0], 'validation', val_df.shape[0])

    train_gen = flow_from_dataframe(train_idg, train_df, path_col='path', y_cols=y_cols,
                                    target_size=img_size,
                                    color_mode='rgb', batch_size=batch_size_train)

    val_gen = flow_from_dataframe(val_idg, val_df, path_col='path', y_cols=y_cols,
                                  target_size=img_size,
                                  color_mode='rgb',
                                  batch_size=batch_size_val)  # we can use much larger batches for evaluation

    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    train_gen = combined_generators(train_gen, train_df[gender_str_col], batch_size_train)
    val_gen = combined_generators(val_gen, val_df[gender_str_col], batch_size_val)

    return train_gen, val_gen, steps_per_epoch, validation_steps


def combined_generators(image_generator, gender, batch_size):
    gender_generator = cycle(batch(gender, batch_size))
    while True:
        nextImage = next(image_generator)
        nextGender = next(gender_generator)
        assert len(nextImage[0]) == len(nextGender)
        yield [nextGender, nextImage[0]], nextImage[1]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def flow_from_dataframe(img_data_gen, in_df, path_col, y_cols, **dflow_args):
    """
    Creates a DirectoryIterator from in_df at path_col with image preprocessing defined by img_data_gen. The labels
    are specified by y_col.

    :param img_data_gen: an ImageDataGenerator
    :param in_df: a DataFrame with images
    :param path_col: name of column in in_df for path
    :param y_cols: list of name of columns in in_df for y values/labels
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
    df_gen.classes = [np.stack(in_df[y_col].values) for y_col in y_cols]
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = base_dir  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    print('flow_from_dataframe() <--')
    return df_gen


def get_chest_dataframe(only_boneage_range):
    img_dir = 'images'
    csv_name = 'sample_labels.csv'
    image_index_col = 'Image Index'

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

    if only_boneage_range:
        chest_df = [x for x in chest_df if x[
            class_str_col_chest] <= 12 * 20]  # delete all entries from set which are not in boneage dataset age range

    return chest_df


def get_boneage_dataframe():
    csv_name = 'boneage-training-dataset.csv'
    img_dir = 'boneage-training-dataset'
    image_index_col = 'id'

    boneage_df = pd.read_csv(os.path.join(boneage_dataset_dir, csv_name))
    boneage_df['path'] = boneage_df[image_index_col].map(
        lambda x: os.path.join(boneage_dataset_dir, img_dir, '{}.png'.format(x)))  # create path from id
    boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
    print('boneage', boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
    boneage_df[gender_str_col_boneage] = boneage_df[gender_str_col_boneage].map(
        lambda x: np.array([1]) if x else np.array([0]))  # map boolean values to 1 and 0

    return boneage_df
