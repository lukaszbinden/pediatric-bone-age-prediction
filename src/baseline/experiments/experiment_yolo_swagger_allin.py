from data_preparation import get_gen
from keras import backend
from model import get_model
from training import train
from testing import test
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import global_hyperparams as hp

# Hyperparameters
NUM_EPOCHS = hp.NUM_EPOCHS
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
LOSS = 'mae'
OPTIMIZER = Adam()
IMG_SIZE = (299, 299)


def execute():
    """
    Transfer learning on Imagenet and then Chest XRays vs. only on Imagenet
    :return:
    """
    train_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

    val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

    train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest = get_gen(train_idg, val_idg,
                                                                                            IMG_SIZE, BATCH_SIZE_TRAIN,
                                                                                            BATCH_SIZE_VAL,
                                                                                            'chest_boneage_range',
                                                                                            disease_enabled=False)
    train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage = get_gen(train_idg, val_idg,
                                                                                                    IMG_SIZE,
                                                                                                    BATCH_SIZE_TRAIN,
                                                                                                    BATCH_SIZE_VAL,
                                                                                                    'boneage',
                                                                                                    disease_enabled=False)

    model = get_model(model='winner', gender_input_enabled=True, age_output_enabled=True, disease_enabled=False,
                      pretrained='imagenet')

    LEARNING_RATE = 1e-3
    OPTIMIZER = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    if CHEST:
        NUM_TRAINABLE_LAYERS = 30
        history = train(train_gen_chest, val_gen_chest, steps_per_epoch_chest,
                        validation_steps_chest, model,
                        OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS,
                        finetuning=False,
                        num_trainable_layers=NUM_TRAINABLE_LAYERS)
        LEARNING_RATE = 1e-4
        OPTIMIZER = SGD(lr=LEARNING_RATE)

    history = train(train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage, model,
                    OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS, finetuning=True,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS)

    print('Boneage dataset (final) history:', history.history)

    test(model)

    backend.clear_session()


if __name__ == '__main__':
    NUM_TRAINABLE_LAYERS = 100
    CHEST = True
    execute()
    CHEST = False
    execute()
