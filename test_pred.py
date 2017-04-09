#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:11:12 2017

@author: mathew
"""
import numpy as np, os, csv, cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K


os.chdir('/Data/cerv_cancer')

#ifile = open('img_key_test.csv','r')

img_list = []
pred_model = load_model('./output/model.h5')
#type_prob = []
with open('img_key_test.csv','r') as ifile:
  reader = csv.reader(ifile)
  for line in reader:
#    print(line[0])
    img = cv2.cvtColor(cv2.imread(line[0]).astype('float32'), cv2.COLOR_BGR2Lab)
    img_list.append(img)

type_prob_full = pred_model.predict(np.array(img_list))
    
ifile.close()

