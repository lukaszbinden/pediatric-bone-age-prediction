from Project.src.baseline.experiments.data_preparation import get_gen
from Project.src.baseline.experiments.model import get_model
from Project.src.baseline.experiments.training import train
import Project.src.baseline.experiments.global_hyperparams as hp

from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

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
    Chest XRays validate against disease and patient age respectively
    :return:
    """
    train_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

    val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

    train_gen_chest, val_gen_chest, steps_per_epoch_chest, validation_steps_chest = get_gen(train_idg, val_idg,
                                                                                            hp.IMG_SIZE, BATCH_SIZE_TRAIN,
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
    OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    LOSS = 'binary_crossentropy'

    history = train(train_gen_chest, val_gen_chest, steps_per_epoch_chest,
                    validation_steps_chest, chest_model,
                    OPTIMIZER, LOSS, LEARNING_RATE, hp.NUM_EPOCHS,
                    finetuning=False,
                    num_trainable_layers=NUM_TRAINABLE_LAYERS,
                    metrics=['mae'])

    print('Chest dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])

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

    OPTIMIZER = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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

    print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])


if __name__ == '__main__':
    # DISEASE_ENABLED = True
    # AGE_ENABLED = True
    # execute()
    # DISEASE_CLASS_STR_COL = 'Finding Labels'
    # DISEASE_ENABLED = True
    # AGE_ENABLED = False
    # execute()
    DISEASE_CLASS_STR_COL = 'Patient Age'
    DISEASE_ENABLED = False
    AGE_ENABLED = True
    execute()

