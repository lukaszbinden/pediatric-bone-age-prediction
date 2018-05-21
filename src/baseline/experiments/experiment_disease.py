from data_preparation import get_gen
from keras import backend
from model import get_model
from training import train
from testing import test
import global_hyperparams as hp

from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
LOSS = 'mae'
OPTIMIZER = Adam()
NUM_TRAINABLE_LAYERS = 20
DISEASE_STR_COL = 'Finding Labels'


def execute():
    """
    Experiment difference if chest X-rays are validated against disease or patient age
    or both
    :return:
    """
    train_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

    val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

    train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest = get_gen(train_idg, val_idg,
                                                                                            hp.IMG_SIZE,
                                                                                            BATCH_SIZE_TRAIN,
                                                                                            BATCH_SIZE_VAL,
                                                                                            'chest',
                                                                                            age_enabled=AGE_ENABLED,
                                                                                            disease_enabled=DISEASE_ENABLED,
                                                                                            predicted_class_col=DISEASE_CLASS_STR_COL)
    train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage, validation_steps_boneage = get_gen(train_idg, val_idg,
                                                                                                    hp.IMG_SIZE,
                                                                                                    BATCH_SIZE_TRAIN,
                                                                                                    BATCH_SIZE_VAL,
                                                                                                    'boneage',
                                                                                                    age_enabled=True,
                                                                                                    disease_enabled=False)

    chest_model = get_model(model='winner',
                            gender_input_enabled=True,
                            age_output_enabled=AGE_ENABLED,
                            disease_enabled=DISEASE_ENABLED,
                            pretrained='imagenet')

    # OPTIMIZER = Adam(lr=1e-3)
    if DISEASE_ENABLED and not AGE_ENABLED:
        OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        LOSS = hp.LOSS_CLASSIFICATION
        METRIC = hp.METRIC_CLASSIFICATION
    elif AGE_ENABLED and not DISEASE_ENABLED:
        OPTIMIZER = hp.OPTIMIZER_ADAM_DEFAULT
        LOSS = hp.LOSS_DEFAULT
        METRIC = hp.METRIC

    history = train(train_gen_chest, val_gen_chest, steps_per_epoch_chest,
                    validation_steps_chest, chest_model,
                    OPTIMIZER, LOSS, LEARNING_RATE, hp.NUM_EPOCHS,
                    finetuning=False,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS,
                    metrics=METRIC)

    print('Chest dataset (final) history:', history.history)

    if DISEASE_ENABLED and not AGE_ENABLED:
        # now build new model for age prediction
        boneage_model = get_model(model='winner',
                                  gender_input_enabled=True,
                                  age_output_enabled=True,
                                  disease_enabled=False,
                                  pretrained='imagenet')

        # save all weights from chest learning
        weights_chest_learning = [layer.get_weights() for layer in chest_model.layers]
        # now do transfer learning: transfer weights (except last layer)
        num_layers = len(weights_chest_learning)
        num_layers_transfer = num_layers - 1
        print('transfer ', num_layers_transfer, 'of total of ', num_layers)
        for i in range(num_layers_transfer):
            boneage_model.layers[i].set_weights(weights_chest_learning[i])
    elif AGE_ENABLED and not DISEASE_ENABLED:
        boneage_model = chest_model

    OPTIMIZER = hp.OPTIMIZER_FINETUNING
    LOSS = 'mae'
    history = train(train_gen_boneage, val_gen_boneage, steps_per_epoch_boneage,
                    validation_steps_boneage, boneage_model,
                    OPTIMIZER,
                    LOSS,
                    LEARNING_RATE,
                    hp.NUM_EPOCHS,
                    finetuning=True,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS,
                    metrics=hp.METRIC)

    print('Boneage dataset (final) history:', history.history)

    test(boneage_model)

    # backend.clear_session()



if __name__ == '__main__':
    # DISEASE_ENABLED = True
    # AGE_ENABLED = True
    # execute()
    DISEASE_CLASS_STR_COL = 'Finding Labels'
    DISEASE_ENABLED = True
    AGE_ENABLED = False
    execute()
    DISEASE_CLASS_STR_COL = 'Patient Age'
    DISEASE_ENABLED = False
    AGE_ENABLED = True
    execute()
