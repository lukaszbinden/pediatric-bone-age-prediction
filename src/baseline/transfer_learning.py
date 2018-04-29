'''
Idea:
Start with inceptionnet/resnet/vgg pretrained on imagenet
add own classifier on top of it: flatten, dense, relu, dropout, dense, sigmoid
train own classifier, freeze feature extractor net
train own classifier and last convnet block
(train with small learning rate and sgd to not destroy previously learned features)

Be happy and hopefully win the competition ;)


'''