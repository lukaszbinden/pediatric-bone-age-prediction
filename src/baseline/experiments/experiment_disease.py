from data_preparation import get_gen
from model import get_model
from training import train
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
LOSS = 'mae'
OPTIMIZER = Adam()
NUM_TRAINABLE_LAYERS = 10
IMG_SIZE = (299, 299)


def execute():
    """
    Chest XRays validate against disease and patient age respectively
    :return:
    """
    train_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

    val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

    train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest = get_gen(train_idg, val_idg,
                                                                                            IMG_SIZE, BATCH_SIZE_TRAIN,
                                                                                            BATCH_SIZE_VAL,
                                                                                            'chest_boneage_range',
                                                                                            disease_enabled=DISEASE_ENABLED)
    train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage = get_gen(train_idg, val_idg,
                                                                                                    IMG_SIZE,
                                                                                                    BATCH_SIZE_TRAIN,
                                                                                                    BATCH_SIZE_VAL,
                                                                                                    'boneage',
                                                                                                    disease_enabled=False)

    model = get_model(model='winner', gender_input_enabled=True, age_output_enabled=AGE_ENABLED, disease_enabled=DISEASE_ENABLED,
                      pretrained='imagenet')

    OPTIMIZER = Adam(lr=1e-3)

    history = train(train_gen_chest, val_gen_chest, steps_per_epoch_chest,
                    validation_steps_chest, model,
                    OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS,
                    finetuning=False,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS)

    OPTIMIZER = SGD(lr=1e-4)

    history = train(train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage, model,
                    OPTIMIZER, LOSS, LEARNING_RATE, NUM_EPOCHS, finetuning=True,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS)

    print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])


if __name__ == '__main__':
    DISEASE_ENABLED = True
    AGE_ENABLED = True
    execute()
    DISEASE_ENABLED = True
    AGE_ENABLED = False
    execute()
    DISEASE_ENABLED = False
    AGE_ENABLED = True
    execute()
