# coding: utf-8
#! /usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse, sys, os, errno
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import keras as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from unet_128_model_row_column import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
set_session(tf.Session(config=config))
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects


#540条
ic_shape = {}
with h5py.File('known/known.h5') as f:
    for i in tqdm(range(540)):
        ic_shape[i] = f['known'][f['start'][i]:f['end'][i]]
    icshape = np.array([value for (key,value) in ic_shape.iteritems()])
    name = f['name'][:]

model_path = 'output/newunet_row_col_mse_1.31_pick.hdf5'
def Model(model_path):
    #为转成十六通道的图片预测shape做准备
    model = UNET_128()
    optim = Adam()
    model.compile(optimizer=optim, loss=CrossEntropyLoss(model,10), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])
    loss=CrossEntropyLoss(model,10)
    model = load_model(model_path,custom_objects = {"CrossEntropyLoss": loss,\
                       'binary_accuracy_with_nan':binary_accuracy_with_nan,\
                       'binary_crossentropy_with_nan':binary_crossentropy_with_nan,\
                       'MSE':MSE(model)})
    return model
model = Model(model_path)

save_path = 'acc_1.31_pick'

def calculate_acc(array,true_score):
    '''
        an array of shape (length)*128
        对齐并且求中间每个位置的平均
        return shape*shape+128
        '''
    shape = array.shape[0]
    new = np.ndarray([shape,shape+128])
    for i in range(shape):
        new[i] = np.concatenate((np.concatenate((np.zeros(i),array[i])),np.zeros(shape-i)))
    score_vector = np.sum(new,axis = 0)[64:-64].astype('float')  #vector  shape
    #这里要分情况！ 长度 64 128为界限  count_vector不一样
    #64以下 每个位置被算长度次
    if shape <=64:
        count_vector = np.repeat(shape,shape)
    if shape >=128:
        count_vector = np.concatenate((np.concatenate((np.arange(65,129),np.repeat(128,shape-128))),~np.arange(65,129) +193))
    if shape >64 and shape <128:
        count_vector = np.concatenate((np.concatenate((np.arange(65,shape+1),np.repeat(shape,128-shape))),~np.arange(65,shape+1) +shape+1 +65))
    score = score_vector/count_vector
    for i in range(shape):
        if score[i] <0.5:
            score[i] = 0
        else:
            score[i] = 1
    acc = float(np.where(np.abs(score-true_score) ==0)[0].shape[0])/float(shape)
    return acc
acc= {}
predict_result = {}
with h5py.File('known/pictures_540') as f:
    for i in tqdm(range(540)):
        images = f[str(i)][:]
        predict_result[i] = model.predict(images)[:,:128]
        acc[i] = calculate_acc(predict_result[i],icshape[i])
        with h5py.File('known/accuracy/'+save_path) as t:
            t.create_dataset(str(i),data = acc[i])
acc = np.array([val for (key,val) in acc.iteritems()])
np.savetxt('known/accuracy/'+save_path+'.txt',acc)

category = np.ndarray([540]).astype('S')
for i in range(540):
    category[i] = name[i].split('_')[0]
name_list = np.unique(category,)
acc_cate = {}
for j in range(10):
    acc_cate[name_list[j]] = []
    for i in range(540):
        if category[i] ==name_list[j]:
            acc_cate[name_list[j]].append(acc[i])
table = pd.DataFrame([10])
for i in range(10):
    table[i] =  sum(acc_cate[name_list[i]])/len(acc_cate[name_list[i]])
table = table.T
table = table.set_index(name_list)
table.columns = ['2d_model']
table['2d_model_restrict'] = np.array(pd.read_csv('known/accuracy/acc_1.30_3')['2d_model'])
table['dense'] = ['0.597','0.684','0.628','0.634','0.587','0.524','0.578','0.563','0.555','0.602']
table = table.round(3)
table.to_csv('known/accuracy/'+save_path)
