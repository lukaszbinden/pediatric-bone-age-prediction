from data_preparation import get_gen, get_boneage_dataframe, flow_from_dataframe, combined_generators
from keras.preprocessing.image import ImageDataGenerator
import global_hyperparams as hp


def test_w_classification(boneage_model):
    test_impl(boneage_model, True)


def test(boneage_model):
    test_impl(boneage_model, False)


def test_impl(boneage_model, classification):
    print('test -->')
    test_idg = ImageDataGenerator()
    test_df = get_boneage_dataframe('boneage-test-dataset', 'Case ID', classification)
    test_gen = flow_from_dataframe(test_idg, test_df, path_col='path', y_col='boneage', target_size=hp.IMG_SIZE,
                                   color_mode='rgb', batch_size=202)  # one big batch

    test_X, test_Y = next(combined_generators(test_gen, test_df['male'], None, True, False, 202))

    scores = boneage_model.evaluate(test_X, test_Y, verbose=1)
    print('Boneage test dataset scores:', scores)
    print(boneage_model.metrics)
    print('test <--')


