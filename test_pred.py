#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:11:12 2017

@author: mathew
"""
import tensorflow as tf
import numpy as np, os, csv, cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

os.chdir('/Data/cerv_cancer')

#ifile = open('img_key_test.csv','r')

img_list = []
pred_model = load_model('./output/model.h5')
#type_prob = []
with open('img_key_test.csv','r') as ifile:
  reader = csv.reader(ifile)
  for line in reader:
    img = cv2.imread(line[0]).astype('float32')
    img_list.append(img)

type_prob = pred_model.predict(np.array(img_list))
    
ifile.close()

