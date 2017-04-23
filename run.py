import sys
import os
import numpy as np

import cv2
import json

import retinex

data_path = '/Data/cerv_cancer/Data/train/Type_1'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

with open('/Data/cerv_cancer/img_key_train.csv','r') as csvfile:
  if img_name == '.gitkeep':
    continue
  
  reader = csv.reader(csvfile)
  for row in reader:
       
#    file_path = os.path.join(data_path, img_name)
    img = cv2.imread(row[0])

    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
   
    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )

    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']        
    )    
    
#    cv2.imwrite(file_path.replace('retinex/data','retinex/data2').replace('.jpg','_msrcr.jpg'), img_msrcr)
#    cv2.imwrite(file_path.replace('retinex/data','retinex/data2').replace('.jpg','_amsrcr.jpg'), img_amsrcr)
    cv2.imwrite(file_path.replace('Data/train','Data/train_r'), img_msrcp)
#    print(np.pi)
#    cv2.imwrite()
#
#    shape = img.shape
#    cv2.imshow('Image', img)
#    cv2.imshow('retinex', img_msrcr)
#    cv2.imshow('Automated retinex', img_amsrcr)
#    cv2.imshow('MSRCP', img_msrcp)
#    cv2.waitKey()
