import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imsave



target_width = 384
target_heigth = 384
target_ratio = target_width/target_heigth

source_folder = "/home/guy/jmcs-atml-bone-age-prediction/boneage-training-dataset/"
target_folder = "/home/guy/jmcs-atml-bone-age-prediction/DatasetTransform2/"

sizes = []
sum_ratio = 0
nbriter = 0



for i in range(1378, 1450): #15610
    try:
        img = Image.open(source_folder + str(i) + ".png")
        img = np.asarray(img)
        
        heigth, width = img.shape
        ratio = width/heigth
        
        if heigth>width:
            #------------------------------------------------------------
            # Resize keeping width
            #------------------------------------------------------------
            new_heigth = int(img.shape[0]/(img.shape[1]/target_heigth))
            img_resized = resize(img, (new_heigth, target_width))
            heigth_resized, width_resized = img_resized.shape
            y1 = int((heigth_resized-target_heigth)/2)
            y2 = y1+target_heigth
            img_cropped = img_resized[y1:y2,0:target_width] # height, width
            imsave(target_folder + str(i) + '.png', img_cropped)
        else:
            #------------------------------------------------------------
            # Resize keeping heigth
            #------------------------------------------------------------
            new_width = int(img.shape[1]/(img.shape[0]/target_width))
            img_resized = resize(img, (target_heigth, new_width))
            heigth_resized, width_resized = img_resized.shape
            x1 = int((width_resized-target_width)/2)
            x2 = x1+target_width
            img_cropped = img_resized[0:target_heigth,x1:x2]
            imsave(target_folder + str(i) + '.png', img_cropped)
        
        
    except:
        print("file doesn't exist")

#print("ratio average: " + str(sum_ratio/nbriter))