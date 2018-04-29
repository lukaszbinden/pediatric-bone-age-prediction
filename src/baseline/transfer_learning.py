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

conv_base_model = InceptionResNetV2(include_top=True,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=1000)

