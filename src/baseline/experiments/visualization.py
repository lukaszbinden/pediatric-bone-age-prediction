import matplotlib.pyplot as plt


def plot(filename, history):
    if 'mae' in history.history.keys():
        print("plotting mae")
        #  "Mean absolute error"
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('model mean absolute error')
        plt.ylabel('mean absolute error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("plots/" + filename + "_mae.jpg")
    if 'acc' in history.history.keys():
        print("plotting acc")
        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("plots/" + filename + "_acc.jpg")
    if 'loss' in history.history.keys():
        print("plotting loss")
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("plots/" + filename + "_loss.jpg")
