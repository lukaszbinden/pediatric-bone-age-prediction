import numpy as np
import pandas as pd
import os

base_dir = '/var/tmp/studi5/boneage/'
base_datasets_dir = base_dir + '/datasets/'

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


def get_chest_dataframe(dir_name):
    base_chest_dir = base_datasets_dir + dir_name
    image_index_col = 'Image Index'
    class_str_col = 'Patient Age'

    chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'), usecols=[image_index_col, class_str_col])
    chest_df[class_str_col] = [int(x[:-1] if type(x) == str and x[-1] == 'Y' else x) * 12 for x in chest_df[class_str_col]]  # parse Year Patient Age to Month age
    chest_df['path'] = chest_df[image_index_col].map(
        lambda x: os.path.join(base_chest_dir, 'images', x))  # create path from id
    chest_df['exists'] = chest_df['path'].map(os.path.exists)
    print(chest_df['exists'].sum(), 'images found of', chest_df.shape[0], 'total')
    # chest_df['chest_category'] = pd.cut(chest_df[class_str], 10)
    return chest_df
