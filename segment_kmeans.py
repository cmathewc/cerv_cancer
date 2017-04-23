#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:29:20 2017

@author: mathew
"""
import os, cv2, csv, numpy as np
import matplotlib.image as mpimg
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import multiprocessing

def compute_feats(image, kernels):
#    feats = np.zeros((len(kernels), 2), dtype=np.double)
    filtered = []
    for k, kernel in enumerate(kernels):
      filtered.append(ndi.convolve(image, kernel, mode='wrap'))
      
    return filtered
        
    
        

def hist_eq(img):
  
  img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL) #Convert RGB to HSV
  img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:,:,2]) #Perform Hist Eq. on value channel
  return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)
  
def process_image(img_pth):
  
  sqrt_2 = np.sqrt(2)

# Build gabor kernels
  gabor_kernels= []
  for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
      for frequency_multiplier in range(2, 10, 1):
        frequency = frequency_multiplier*sqrt_2
        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma))
        gabor_kernels.append(kernel)
  
  img = cv2.resize(cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB), dsize=(512, 683))
  hist_eq_img = hist_eq(img)
  
  filt_op = [];
  for i in range(3):
    filt_op.append(np.array(compute_feats(hist_eq_img[:,:,i].astype('float32'), gabor_kernels)))
    
  np.save(img_pth.replace('Data/train','Data/train_gabor').replace('.jpg',''),np.array(filt_op))
#  return(np.array(filt_op))
  
if __name__ == '__main__':
  os.chdir("/Data/cerv_cancer") 
  img_path = [];
  jobs = [];
  with open('img_key_train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    pool = multiprocessing.Pool(6)
    for row in reader:
      img_path.append(row[0])
    pool.map(process_image, img_path)
        
    pool.close()
    pool.terminate()
    pool.join()
#    i = 0;
#        for i in range(len(img_path)):
#          full_img = cv2.imread(img_path[i])
#          dsample_img = hist_eq_crop_img(full_img)
#          cv2.imwrite(img_path[i].replace('Data/train','Data/train_rect'), dsample_img)
            
  csvfile.close()
  
#os.chdir("/Data/cerv_cancer") 
#img_path = [];
#jobs = [];
#
#
#
#with open('img_key_train.csv','r') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
#      process_image(row[0], kernels)
      
      