# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np, os, cv2, csv, pickle

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
  img_data = []#np.zeros((1,512,512, 3)).astype('float32')
  img_label = []#np.int(0)
  
  csvFile = './img_key.csv'
  
  with open(csvFile) as fid:
    csvReader = csv.reader(fid)
    for row in csvReader:
      img_data.append(cv2.imread(row[0]).astype('float32'))
      img_label.append(np_utils.to_categorical(np.int(row[1])-1, num_classes =3))
      ct += 1
      if np.remainder(ct, 100) == 0:
        print('Iteration_{}'.format(ct))
        
        
  #img_label_cat = np_utils.to_categorical(img_label[1:] - 1, num_classes = 3)
  #img_data = img_data[1:,:,:,:]
      
  return img_data, img_label, ct
    
(img, label, ct) = load_data()    

np.savez('img_data',img_data = np.array(img), label_data = np.array(label))
#
#with open('img_data.txt','wb') as tf:
#  pickle.dump(img, tf)
#  pickle.dump(label, tf)
#  
#tf.close()