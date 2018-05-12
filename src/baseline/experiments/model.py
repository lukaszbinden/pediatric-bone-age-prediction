from keras import Input, Model
from keras.applications import InceptionV3, InceptionResNetV2, VGG16
from keras.layers import Flatten, Dense, concatenate, AveragePooling2D, BatchNormalization, LocallyConnected2D, Conv2D, \
    multiply, GlobalAveragePooling2D, Lambda, Dropout
import numpy as np


def get_model(model='baseline', gender_enabled=True, pretrained='imagenet'):
    """

    :param model: 'baseline', 'own' or 'winner
    :param gender_enabled: True or False
    :param pretrained: 'imagenet' or None
    :return:
    """
    input_gender = Input(shape=(1,), name='input_gender')
    input_img = Input(shape=(299, 299, 3), name='input_img')

    conv_base = get_conv_base(input_img, model, pretrained)
    if gender_enabled:
        feature = concatenate([get_gender(input_gender), conv_base], axis=1)
    else:
        feature = conv_base

    output = get_classifier(feature)

    return Model(inputs=[input_gender, input_img], outputs=output)


def get_conv_base(input_img, model, pretrained):
    if model == 'baseline':
        return get_baseline(input_img, pretrained)
    elif model == 'own':
        return get_own(input_img, pretrained)
    elif model == 'winner':
        return get_winner(input_img, pretrained)


def get_winner(input_img, pretrained):
    base = InceptionV3(input_tensor=input_img, input_shape=(299, 299, 3), include_top=False, weights=pretrained)
    feature_img = base.get_layer(name='mixed10').output
    # feature_img = AveragePooling2D((2,2), name='ave_pool_fea')(feature_img)
    # feature_img = Flatten()(feature_img)
    # feature_img = GlobalAveragePooling2D()(feature_img)
    feature_img = AveragePooling2D((2, 2))(feature_img)
    feature_img = Flatten()(feature_img)

    return feature_img


def get_own(input_img, pretrained):
    base = InceptionResNetV2(include_top=True,
                             weights=pretrained,
                             input_tensor=input_img,
                             # input_shape=t_x.shape[1:],
                             # pooling=None,
                             # classes=1000
                             )
    feature_img = base.get_layer(name='mixed10').output
    # feature_img = AveragePooling2D((2,2), name='ave_pool_fea')(feature_img)
    # feature_img = Flatten()(feature_img)
    # feature_img = GlobalAveragePooling2D()(feature_img)
    feature_img = AveragePooling2D((2, 2))(feature_img)
    feature_img = Flatten()(feature_img)

    return feature_img


def get_baseline(input_img, pretrained):
    base_pretrained_model = VGG16(input_tensor=input_img, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = False
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(input_img)

    bn_features = BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on and off

    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = LocallyConnected2D(1,
                                    kernel_size=(1, 1),
                                    padding='valid',
                                    activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.25)(Dense(1024, activation='relu')(gap_dr))
    return Dense(1, activation='linear')(dr_steps)  # linear is what 16bit did


def get_classifier(feature):
    output = Dense(1000, activation='relu')(feature)
    output = Dense(1000, activation='relu')(output)
    output = Dense(1)(output)
    return output


def get_gender(input_gender):
    feature_gender = Dense(32, activation='relu')(input_gender)
    return feature_gender
