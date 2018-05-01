'''
Idea:
Start with inceptionnet/resnet/vgg pretrained on imagenet: https://keras.io/applications/
add own classifier on top of it: flatten, dense, relu, dropout, dense, sigmoid
train own classifier, freeze feature extractor net
train own classifier and last convnet block
(train with small learning rate and sgd to not destroy previously learned features)

Be happy and hopefully win the competition ;)


'''
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.model_selection import train_test_split

conv_base_model = InceptionResNetV2(include_top=True,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=1000)

base_chest_dir = '/var/tmp/studi5/boneage/nih-chest-xrays/'
age_df = pd.read_csv(os.path.join(base_chest_dir, 'sample_labels.csv'))
print(age_df)

#raw_train_df, valid_df = train_test_split(age_df, test_size=0.2, random_state=2018, stratify=age_df['boneage_category'])
#print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
