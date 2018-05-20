# global hyperparameters to ensure
# that all experiements are using
# the same values to enable a uniform
# training setup and eventually a
# fair performance comparison.
from keras.optimizers import Adam, SGD

NUM_EPOCHS = 2
IMG_SIZE = (299, 299)
METRIC = ['mae']
LOSS_DEFAULT = 'mae'
LOSS_CLASSIFICATION = 'categorical_crossentropy'
LEARNING_RATE_DEFAULT = 0.001
OPTIMIZER_ADAM_DEFAULT = Adam(lr=LEARNING_RATE_DEFAULT, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
OPTIMIZER_SGD_DFEAULT = SGD(lr=1e-4)
OPTIMIZER_FINETUNING = OPTIMIZER_SGD_DFEAULT
