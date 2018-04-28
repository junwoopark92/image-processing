import pandas as pd
import numpy as np
from PIL import Image
import sys
import re
import time
import os

import base64
import requests
import json
from io import BytesIO
from IPython.display import HTML

import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

p = re.compile("[a-zA-z]{5}")

def get_image_batch_from_file(files):
    p = re.compile("[a-zA-z]{5}")
    return_list = []
    for i, path in enumerate(files):
        #print(i, path)
        return_list.append(np.array(image2feature(path,i,p)).reshape(224,224,3))
    
    return return_list

def image2feature_from_url(img_url):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224,224),Image.ANTIALIAS)
        img = img.convert("RGB")
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img,axis=0)
        img_data = img_data.copy()
        img_data = img_data.astype("float32")
        img_data = preprocess_input(img_data)
    except Exception as e:
        print('bug: ',e,'url: ',img_url)
        img_data = np.zeros((1,224,224,3))
    
    return img_data.tolist()


def image2feature_from_file(img_path,index, p):

#     if index%100 == 0:
#         print(index)
    
    img_path = img_path.replace('HTTP://I.011ST.COM','')
    rm = p.findall(img_path)
    if len(rm) == 1:
        img_path = img_path.replace(rm[0]+'/','').replace(' ','')
    try:
        img_path = '/data1/upload'+img_path
        img = Image.open(img_path).convert('RGB').resize((224,224),Image.ANTIALIAS)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img,axis=0)
        img_data = img_data.copy()
        img_data = img_data.astype('float32')
        img_data = preprocess_input(img_data)
    except Exception as e:
        print('bug:',e, img_path)
        img_data = np.zeros((1,224,224,3))

    return img_data.tolist()

def get_image_batch_from_url(urls):
    return_list = []
    for i, url in enumerate(urls):
        #print(i, url)
        return_list.append(np.array(image2feature_from_url(url)).reshape(224,224,3))
    
    return return_list

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, files, batch_size):
        
        self.batch_size = batch_size
        self.files = files
        self.on_epoch_end()
 
    def __len__(self):
        data_size = len(self.files)
        batch_size = self.batch_size
        step = 0
        if data_size%batch_size == 0:
            step = int(data_size/batch_size)
        else:
            step = int(data_size/batch_size) + 1
      
        return step

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        files_temp = [self.files[k] for k in indexes]
        X = self.__data_generation(files_temp)
        
        return X
    
    def on_epoch_end(self):
        print('epoch_end call')
        self.indexes = np.arange(len(self.files))
        
    def __data_generation(self, files_temp):
        X = np.array(get_image_batch_from_url(files_temp))
        return X