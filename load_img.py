# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np, os, cv2, csv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

os.chdir("/Data/cerv_cancer") 

def load_data():
  path_1 = os.path.join(os.getcwd(),'Data','train_small2','Type_1')
  path_2 = os.path.join(os.getcwd(),'Data','train_small2','Type_2')
  path_3 = os.path.join(os.getcwd(),'Data','train_small2','Type_3')
  
  ct = 0
  img_data = np.zeros((1,512,512, 3)).astype('float32')
  img_label = np.int(0)
  
  csvFile = './img_key.csv'
  
  with open(csvFile) as fid:
    csvReader = csv.reader(fid)
    for row in csvReader:
      img_data = np.vstack((img_data, cv2.imread(row[0]).reshape(1, 512, 512, 3).astype('float32')))
      img_label = np.hstack((img_label, np.int(row[1])))
      ct += 1
      if np.remainder(ct, 100) == 0:
        print('Iteration_{}'.format(ct))
        
        
  img_label_cat = np_utils.to_categorical(img_label[1:] - 1, num_classes = 3)
  img_data = img_data[1:,:,:,:]
      
  return img_data, img_label_cat, ct
    
(img, label, ct) = load_data()    

np.savez('img_data',img_data = img, label_data = label)