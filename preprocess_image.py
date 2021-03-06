#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:57:15 2017

@author: mathew

Preprocess images used for cervical cancer screening challenge
"""
import os, cv2, csv
import multiprocessing
import numpy as np

import json

import retinex


os.chdir("/Data/cerv_cancer") 

file_path = os.path.join(os.getcwd(), 'Data','train', 'Type_1', '0.jpg')
img = cv2.imread(file_path)
#
#
cv2.namedWindow('dst_rt', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('dst_rt', 512, 512)
    
##def crop_image(img):
def crop_black_bkgd(img):
#    #Invert the image to be white on black for compatibility with findContours function.
#    #Code from https://www.kaggle.com/cpruce/intel-mobileodt-cervical-cancer-screening/cervix-image-segmentation
#
    try:
      img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  #    cv2.imshow('dst_rt', imgray)
  #    cv2.waitKey(0)
      #Binarize the image and call it thresh.
      ret, thresh = cv2.threshold(img_hsv[:,:,2], 127, 255, cv2.THRESH_BINARY) #Threshold on the V channel to avoid errors in dark images
  #    cv2.imshow('dst_rt', thresh)
  #    cv2.waitKey(0)
      #Find all the contours in thresh. In your case the 3 and the additional strike
      im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      #Calculate bounding rectangles for each contour.
      rects = [cv2.boundingRect(cnt) for cnt in contours]
      
      #Calculate the combined bounding rectangle points.
      top_x = min([x for (x, y, w, h) in rects])
      top_y = min([y for (x, y, w, h) in rects])
      bottom_x = max([x+w for (x, y, w, h) in rects])
      bottom_y = max([y+h for (x, y, w, h) in rects])
      
      #Draw the rectangle on the image
      #    out = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 2)
      crop = img[top_y:bottom_y,top_x:bottom_x]
      return crop #thresh
    except:
      return img
#    
#
def crop_resize_img(img):
    img_crop_resize = cv2.resize(crop_black_bkgd(img), dsize = (512, 512))
    return img_crop_resize

def hist_eq_crop_img(img):
#    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert BGR to HSV
#    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:,:,2]) #Perform Hist Eq. on value channel
#    
#    img_bgr_crop = crop_resize_img(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)) #Reconvert to BGR, crop and resize it

#    Instead of cropping and resizing histogram equalized image, histogram equalize cropped image and then resize
    
    img_crop = crop_black_bkgd(img) # Crop out black background
    img_crop_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV) #Convert BGR to HSV
    img_crop_hsv[:, :, 2] = cv2.equalizeHist(img_crop_hsv[:,:,2]) #Perform Hist Eq. on value channel
    
    disp_image = 0
    
    if disp_image:
                                              
        screen_res = 1280, 720
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)
        cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('dst_rt', window_width, window_height)
        cv2.imshow('dst_rt', img)
        
        cv2.namedWindow('dst_rt2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('dst_rt2', 1024, 1024)
        cv2.imshow('dst_rt2', img_bgr_crop)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#    return img_bgr_crop
#    return cv2.resize(cv2.cvtColor(img_crop_hsv, cv2.COLOR_HSV2BGR), dsize = (768, 1024))
    return cv2.cvtColor(img_crop_hsv, cv2.COLOR_HSV2BGR)
    

def process_image_hist_eq_crop(img_path):
  try:
    full_img = cv2.imread(img_path)
    dsample_img = hist_eq_crop_img(full_img)
    cv2.imwrite(img_path.replace('Data/train','Data/train_process'), dsample_img)
  except:
    print(img_path)
    
def process_image_retinex(img_path):
  
  with open('config.json', 'r') as f:
    config = json.load(f)
    
    
  img = crop_resize_img(cv2.imread(img_path))
  
  img_msrcp = retinex.MSRCP(img, config['sigma_list'], config['low_clip'], config['high_clip'])
  
  cv2.imwrite(img_path.replace('Data/train','Data/train_r'), img_msrcp)
  
  img_msrcp = retinex.MSRCP(hist_eq_crop_img(img), config['sigma_list'], config['low_clip'], config['high_clip'])
  
  cv2.imwrite(img_path.replace('Data/train','Data/train_r2'), img_msrcp)
  
  img_msrcp = hist_eq_crop_img(retinex.MSRCP(img, config['sigma_list'], config['low_clip'], config['high_clip']))
  
  cv2.imwrite(img_path.replace('Data/train','Data/train_r3'), img_msrcp)
  
  
###def read_process_image():
if __name__ == '__main__':
    os.chdir("/Data/cerv_cancer") 
    img_path = [];
    jobs = [];
    with open('img_key_train.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        pool = multiprocessing.Pool(6)
        for row in reader:
          img_path.append(row[0])
#        pool.map(process_image_hist_eq_crop, img_path)
        pool.map(process_image_retinex, img_path)
        
        
        pool.close()
        pool.terminate()
        pool.join()
#    i = 0;
#        for i in range(len(img_path)):
#          full_img = cv2.imread(img_path[i])
#          dsample_img = hist_eq_crop_img(full_img)
#          cv2.imwrite(img_path[i].replace('Data/train','Data/train_rect'), dsample_img)
            
        csvfile.close()
    

#return img
#if __name__ == "__main__":
#    main()
#crop_black_bkgd(img)
enable_disp = 0
if enable_disp:
    
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    
    cv2.namedWindow('dst_rt2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt2', window_width, window_height)
    
    cv2.namedWindow('dst_rt3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt3', window_width, window_height)
    
    # Histogram Equalization
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img_hsv_val = img_hsv[:,:,2]
    
    h_clahe = cv2.createCLAHE(tileGridSize = (1024, 1024))
    img_hsv_val_he = cv2.equalizeHist(img_hsv_val)# 
    #img_hsv_val_he = h_clahe.apply(img_hsv_val)
    img_hsv_he = img_hsv
    img_hsv_he[:, :, 2] = img_hsv_val_he
    img_he = cv2.cvtColor(img_hsv_he, cv2.COLOR_HSV2BGR)
    
    
    
    crop_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    crop_hsv_val = crop_hsv[:,:,2]
    h_clahe = cv2.createCLAHE(tileGridSize = (1024, 1024))
    crop_hsv_val_he = cv2.equalizeHist(crop_hsv_val)# 
    #crop_hsv_val_he = h_clahe.apply(crop_hsv_val)
    crop_hsv_he = crop_hsv
    crop_hsv_he[:, :, 2] = crop_hsv_val_he
    crop_he = cv2.cvtColor(crop_hsv_he, cv2.COLOR_HSV2BGR)
    
    #Try to detect the speculum if it exists. Use Hough circle detector for this purpose. 
    #This way, after eliminating the rest of the image, we will have an approximately 
    #similar resolution of the cervix across the entire dataset.
    
    #img_he_r = img_he[:,:,0]
    #circles_r = cv2.HoughCircles(img_he_r,cv2.HOUGH_GRADIENT,2,512, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)
    ###
    #circles_g = cv2.HoughCircles(img_he[:,:,1],cv2.HOUGH_GRADIENT,1,256, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)
    ##
    #circles_b = cv2.HoughCircles(img_he[:,:,2],cv2.HOUGH_GRADIENT,1,256, param1=50,param2=30,minRadius=np.int(img.shape[1]/2),maxRadius=0)
    #
    #x = np.transpose(circles_r[:,:,0])
    #y = np.transpose(circles_r[:,:,1])
    #r = np.transpose(circles_r[:,:,2])
    #
    #
    #for i in range(len(x)):
    #    cv2.circle(img,(x[i],y[i]), r[i], (0, 255, 0), 4)
        
    #img_raw_he = img
    #for i in range(3):
    ##    img_raw_he[:,:,i] = cv2.equalizeHist(img[:,:,i])# 
    #    img_raw_he[:,:,i] = h_clahe.apply(img[:,:,i])
    img = cv2.imread(file_path)
    cv2.imshow('dst_rt', img)
    cv2.imshow('dst_rt2', img_he)
    cv2.imshow('dst_rt3', crop_he)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

#cv2.destroyAllWindows()