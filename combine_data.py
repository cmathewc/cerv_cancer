#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:43:49 2017

@author: mathew

Code to read data from multiple directories, merge them and create csv file with key.
This enables pooling of data
Creates a file called img_key.csv with details. 
"""

import os, csv

#change drectory to root for this project
os.chdir("/Data/cerv_cancer") 

#print(os.getcwd())
fieldnames = ['Path','Type']

folder_name = 'train_process'

type1_rpath = os.path.join(folder_name, 'Type_1')
type2_rpath = os.path.join(folder_name, 'Type_2')
type3_rpath = os.path.join(folder_name, 'Type_3')

type1_path = os.path.join(os.getcwd(), 'Data', type1_rpath)
type2_path = os.path.join(os.getcwd(), 'Data', type2_rpath)
type3_path = os.path.join(os.getcwd(), 'Data', type3_rpath)

test_path = os.path.join(os.getcwd(), 'Data', 'test_rect')

ofile = open('.'.join(('_'.join(('img_key',folder_name)),'csv')),"w")

data_writer = csv.DictWriter(ofile, fieldnames = fieldnames, delimiter = ',')
###data_writer.writeheader()
for file in os.listdir(type1_path):
    if (file.endswith('.jpeg') or file.endswith('.jpg')):
        data_writer.writerow({'Path': os.path.join(type1_path, file), 'Type':1})
        

for file in os.listdir(type2_path):
    if file.endswith('.jpeg') or file.endswith('.jpg'):
        data_writer.writerow({'Path': os.path.join(type2_path, file), 'Type':2})

for file in os.listdir(type3_path):
    if file.endswith('.jpeg') or file.endswith('.jpg'):
        data_writer.writerow({'Path': os.path.join(type3_path, file), 'Type':3})    
#ct = 0
#for file in os.listdir(test_path):
#    if file.endswith('.jpeg') or file.endswith('.jpg'):
#        data_writer.writerow({'Path': os.path.join(test_path, file), 'Type':0})    
#        ct +=1

ofile.close()