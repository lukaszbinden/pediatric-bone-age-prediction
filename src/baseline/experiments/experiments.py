from data_preparation import get_gen
from model import get_model
from training import train
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
LOSS = 'mae'
OPTIMIZER = Adam()
NUM_TRAINABLE_LAYERS = 10
IMG_SIZE = (299, 299)


def execute():
    # train_idg = ImageDataGenerator(samplewise_center=False,
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

    train_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

    val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

    train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest = get_gen(train_idg, val_idg,
                                                                                            IMG_SIZE, BATCH_SIZE_TRAIN,
                                                                                            BATCH_SIZE_VAL, 'chest')
    train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage = get_gen(train_idg, val_idg,
                                                                                                    IMG_SIZE,
                                                                                                    BATCH_SIZE_TRAIN,
                                                                                                    BATCH_SIZE_VAL,
                                                                                                    'boneage')

    model = get_model('winner', True, 'imagenet')

    history = train(train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest, model,
                    OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS,
                    NUM_TRAINABLE_LAYERS)

    history = train(train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage, model,
                    OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS,
                    NUM_TRAINABLE_LAYERS)

    print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])


if __name__ == '__main__':
    execute()
