import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # showing and rendering figures

# io related
from skimage.io import imread
import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input
##split data into training and validation
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, \
    Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.metrics import mean_absolute_error
from keras.models import load_model

import pickle
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_hist, equalize_adapthist

import ImageSelector as imgsel

Server = False

#/home/guy/jmcs-atml-bone-age-prediction/datasets
#/var/tmp/studi5/boneage/datasets/boneage/
if Server == False:
    base_bone_dir = '/home/guy/jmcs-atml-bone-age-prediction/datasets/'
    path_var_store = '/home/guy/jmcs-atml-bone-age-prediction/variables/'
else:
    base_bone_dir = '/var/tmp/studi5/boneage/datasets/boneage/'
    path_var_store = '/var/tmp/studi5/boneage/variables/'

model_prediction_dir = 'ModelPrediction/'

#-----------------------------------
#LOAD MODEL FOR PREDICTING ACCURACY
#-----------------------------------

model_accuracy = load_model(base_bone_dir + model_prediction_dir+ 'weights-03-0.55.h5')
    
age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))  # read csv
age_df['path'] = age_df['id'].map(lambda x: os.path.join(base_bone_dir, 'boneage-training-dataset','{}.png'.format(x)))  # add path to dictionary


age_df['exists'] = age_df['path'].map(os.path.exists)  # add exists to dictionary
print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')  # print how many images have been found
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')  # convert boolean to string male or female
boneage_mean = age_df['boneage'].mean()
boneage_div = 2 * age_df['boneage'].std()
boneage_mean = 0
boneage_div = 1.0
age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div) # creates classes
age_df.dropna(inplace=True)
age_df.sample(3)
# age_df[['boneage', 'male', 'boneage_zscore']].hist(figsize=(10, 5))
age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)

raw_train_df, valid_df = train_test_split(age_df, test_size=0.2, random_state=2018, stratify=age_df['boneage_category'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

# Balance the distribution in the training set
train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace=True)).reset_index(drop=True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
# train_df[['boneage', 'male']].hist(figsize=(10, 5))

def plotimghist(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.hist(img.flatten(), bins=np.arange(np.min(img),np.max(img),((np.max(img)-np.min(img))/100)))
    plt.show()

def prepro(x):
    
    # ----------------------------------------------------
    # IF ACCURACY PREDICTION IS NOT GOOD ENOUGHT -> REMOVE
    # -----------------------------------------------------
  
    #for i in range(x.shape[2]):
        #img = x[:,:,2]
        #plotimghist(img)
        #img = (img-(np.min(img)))/np.max(img)
        #img = img+(0.5-np.mean(img))
        #img = rescale_intensity(img)  
        #img = equalize_hist(img)$
        #img = equalize_adapthist(img)
        #if i==0:
        #    plotimghist(img)
        #x[:,:,i] = img
    return x

def on_epoch_end_(epoch, logs):
    print("End of Epoch")
    
    
    #----------------Predict output on all data-------------------
    list_data = imgsel.LoadDataList('boneage-training-dataset.csv')#need to be changed
    # Reduced train list
    train_list = dict((k, v) for k, v in list_data.items() if k > 500 and k < 1500)
    #val_list = dict((k, v) for k, v in list_data.items() if k > 300 and k < 350)
    #if epoch == 0:
    img_train, boneage_train, gender_train= imgsel.LoadData2Mem(train_list, 384)
    img_train = imgsel.convert_gray_to_rgb(img_train)
    prediction = bone_age_model.predict(img_train)
    list_png = [x[1][0] for x in train_list.items()]
    list_png_pred = list(zip(list_png,prediction[0]))
    
    #save data
    with open(path_var_store + 'objs2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([list_png_pred], f)

    #if epoch ==0:
        # --------------------------------------------
        # ACCURACY PREDICTOR MODEL
        # --------------------------------------------
        #model_accuracy_predictor.fit()
        #train_list_use = imgsel.LoadDataList('boneage-training-dataset.csv')#need to be changed
        #val_list_use = imgsel.LoadDataList('boneage-test-dataset.csv')

        #num_train_sample = len(train_list_use)
        #num_val_sample = len(val_list_use)

        #img_train, boneage_train, gender_train=imgsel.LoadData2Mem(train_list_use, 500)
        #img_val, boneage_val, gender_val=imgsel.LoadData2Mem(val_list_use, 500)

         
        
IMG_SIZE = (384, 384)  # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(#featurewise_center = True/False
                              samplewise_center=False,
                              #featurewise_std_normalization = True,
                              samplewise_std_normalization=True,
                              #zca_epsilon=True,
                              #zca_whitening=True,
                              rotation_range=20,
                              width_shift_range=0.15,
                              #float = ,
                              #int = ,
                              #shear_range=0.01,
                              zoom_range=0.25,
                              #channel_shift-range = 
                              fill_mode='nearest',
                              #cval =   
                              horizontal_flip=True,
                              vertical_flip=False,
                              #rescale = 
                              #preprocessing_function=prepro,
                              #data_format = 
                              #validation_split
                              height_shift_range=0.15
                              )




def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    # df_gen.
    df_gen._set_index_array()
    df_gen.directory = base_dir  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


train_gen = flow_from_dataframe(core_idg, train_df, path_col='path', y_col='boneage_zscore', target_size=IMG_SIZE, color_mode='rgb', batch_size=32)

valid_gen = flow_from_dataframe(core_idg, valid_df, path_col='path', y_col='boneage_zscore', target_size=IMG_SIZE, color_mode='rgb', batch_size=256)  # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, valid_df, path_col='path', y_col='boneage_zscore', target_size=IMG_SIZE, color_mode='rgb', batch_size=256))  # one big batch

t_x, t_y = next(train_gen)

#show image (barleo01)
'''
for i in range(10):
    img = test_X[i,:,:,0]
    plotimghist(img)
    #mean = np.mean(img)
    #img = img + (100-mean)
    #hist = np.histogram(img.flatten(), bins=np.arange(0,254,5))
    #print(np.amax(hist))   
'''

in_lay = Input(t_x.shape[1:])
base_pretrained_model = VGG16(input_shape=t_x.shape[1:], include_top=False, weights='imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)

bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on and off

attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
attn_layer = LocallyConnected2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.25)(Dense(1024, activation='elu')(gap_dr))
out_layer = Dense(1, activation='linear')(dr_steps)  # linear is what 16bit did
bone_age_model = Model(inputs=[in_lay], outputs=[out_layer])


def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div * in_gt, boneage_div * in_pred)


loss = bone_age_model.compile(optimizer='adam', loss='mse', metrics=[mae_months])

bone_age_model.summary()

weight_path = base_bone_dir + "{}_weights.best.hdf5".format('bone_age')

#checkpoint = ModelCheckpoint(base_bone_dir+'weights-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

lambdacall = LambdaCallback(on_epoch_end = on_epoch_end_) # barleo01

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)  # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat, lambdacall]




history = bone_age_model.fit_generator(train_gen, validation_data=(test_X, test_Y), epochs=1, callbacks=callbacks_list)


with open('/var/tmp/studi5/boneage/git/jmcs-atml-bone-age-prediction/TrainingHistory/history_std_normalization', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
