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
from sklearn.model_selection import train_test_split

conv_base_model = InceptionResNetV2(include_top=True,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=1000)

base_chest_dir = '/var/tmp/studi5/boneage/datasets/nih-chest-xrays/'
chest_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'))
print(chest_df.as_matrix(['Image Index'])[0:3])
print(chest_df.as_matrix(['Patient Age'])[0:3])
print(chest_df.as_matrix(['Image Index', 'Patient Age'])[0:3])

raw_train_df, valid_df = train_test_split(chest_df, test_size=0.2, random_state=2018)  # , stratify=chest_df['Patient Age'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
