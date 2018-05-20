from datetime import datetime

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model


def train(train_gen, val_gen,
          steps_per_epoch,
          validation_steps,
          model,
          optimizer=Adam(),
          loss='mean_absolute_error',
          lr=0.0001,
          num_epochs=100,
          finetuning=False,
          num_trainable_layers=20,
          metrics=['mae']):
    """
    :param train_gen:
    :param val_gen:
    :param steps_per_epoch:
    :param validation_steps:
    :param model:
    :param optimizer:
    :param loss:
    :param lr:
    :param num_epochs:
    :param finetuning: False basically means all layers will be trained
    :param num_trainable_layers:
    :param metrics:
    :return:
    """

    tstart = datetime.now()
    print('training starting --> %s' % str(tstart))

    if finetuning:
        # make last num_trainable_layers of conv layers in model trainable -->
        model.trainable = True
        for layer in model.layers[0:len(model.layers) - num_trainable_layers]:
            layer.trainable = False
        for layer in model.layers[-num_trainable_layers:]:
            layer.trainable = True

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics)  # if two outputs are defined two losses and loss_weights could be defined

    model.summary()  # prints the network structure
    plot_model(model, to_file='model_plot.png', show_shapes=True,
               show_layer_names=True)  # save visualization of model to file

    earlyStopping = EarlyStopping(monitor="val_loss", mode="min",
                                  patience=10)

    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
                                          epsilon=0.0001,
                                          cooldown=5, min_lr=lr * 0.1)

    history = model.fit_generator(train_gen, validation_data=val_gen, epochs=num_epochs, verbose=1,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  callbacks=[earlyStopping, reduceLROnPlateau])  # trains the model

    tend = datetime.now()
    print('training finished: %s' % str((tend - tstart)))

    return history
