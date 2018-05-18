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
    Experiment with different number of freezed layers: between 10 and 100 with step 10 tried
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
                                                                                                    'boneage', disease_enabled=False)

    model = get_model(model='winner', gender_input_enabled=True, age_output_enabled=True, disease_enabled=False,
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
    for num_trainable_layers in range(10, 100, 10):
        NUM_TRAINABLE_LAYERS = num_trainable_layers
        execute()
