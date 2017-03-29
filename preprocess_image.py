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
    
file_path = os.path.join(os.getcwd(), 'Data','train', 'Type_1', '0.jpg')
img = cv2.imread(file_path)


screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt', window_width, window_height)

# Histogram Equalization
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

img_hsv_val = img_hsv[:,:,2]

h_clahe = cv2.createCLAHE(tileGridSize = (64, 64))
img_hsv_val_he = h_clahe.apply(img_hsv_val)
img_hsv_he = img_hsv
img_hsv_he[:, :, 2] = img_hsv_val_he
img_he = cv2.cvtColor(img_hsv_he, cv2.COLOR_HSV2RGB)

#Try to detect the speculum if it exists. Use Hough circle detector for this purpose. 
#This way, after eliminating the rest of the image, we will have an approximately 
#similar resolution of the cervix across the entire dataset.

img_he_r = img_he[:,:,0]
circles_r = cv2.HoughCircles(img_he_r,cv2.HOUGH_GRADIENT,1,256, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)
##
circles_g = cv2.HoughCircles(img_he[:,:,1],cv2.HOUGH_GRADIENT,1,256, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)
#
circles_b = cv2.HoughCircles(img_he[:,:,2],cv2.HOUGH_GRADIENT,1,256, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)

x = np.transpose(circles_r[:,:,0])
y = np.transpose(circles_r[:,:,1])
r = np.transpose(circles_r[:,:,2])


for i in range(len(x)):
    cv2.circle(img,(x[i],y[i]), r[i], (0, 255, 0), 4)
    
cv2.imshow('dst_rt', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#cv2.destroyAllWindows()