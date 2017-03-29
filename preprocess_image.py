#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:57:15 2017

@author: mathew

Preprocess images used for cervical cancer screening challenge
"""
import os, cv2
import numpy as np
os.chdir("/Data/cerv_cancer") 

#def preproces_image(file_path):
    
#if __name__ == "__main__":
#    main()
    
file_path = os.path.join(os.getcwd(), 'Data','train', 'Type_1', '47.jpg')
img = cv2.imread(file_path)


screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt', window_width, window_height)

#cv2.imshow('dst_rt', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

img_hsv_val = img_hsv[:,:,2]

#cv2.imshow('dst_rt', img_hsv_val)
#cv2.waitKey(0)

h_clahe = cv2.createCLAHE(tileGridSize = (64, 64))
img_hsv_val_he = h_clahe.apply(img_hsv_val)
img_hsv_he = img_hsv
img_hsv_he[:, :, 2] = img_hsv_val_he
img_he = cv2.cvtColor(img_hsv_he, cv2.COLOR_HSV2RGB)

cv2.imshow('dst_rt', img_he)
cv2.waitKey(0)

cv2.destroyAllWindows()