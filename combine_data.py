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

type1_rpath = os.path.join('additional', 'Type_1')
type2_rpath = os.path.join('additional', 'Type_2')
type3_rpath = os.path.join('additional', 'Type_3')

type1_path = os.path.join(os.getcwd(), 'Data', type1_rpath)
type2_path = os.path.join(os.getcwd(), 'Data', type2_rpath)
type3_path = os.path.join(os.getcwd(), 'Data', type3_rpath)

ofile = open('img_key_additional.csv',"w")
data_writer = csv.DictWriter(ofile, fieldnames = fieldnames, delimiter = ',')
#data_writer.writeheader()
for file in os.listdir(type1_path):
    if (file.endswith('.jpeg') or file.endswith('.jpg')):
        data_writer.writerow({'Path': os.path.join(type1_path, file), 'Type':1})
        

for file in os.listdir(type2_path):
    if file.endswith('.jpeg') or file.endswith('.jpg'):
        data_writer.writerow({'Path': os.path.join(type2_path, file), 'Type':2})

for file in os.listdir(type3_path):
    if file.endswith('.jpeg') or file.endswith('.jpg'):
        data_writer.writerow({'Path': os.path.join(type3_path, file), 'Type':3})    