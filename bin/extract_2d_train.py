#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import h5py
import seaborn as sns
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file')
parser.add_argument('-c', dest='count')
args = parser.parse_args()

def convert_to_image(x):
    m = (np.repeat(x, 4, axis=1)[np.newaxis, :, :]*np.tile(x, 4)[:, np.newaxis, :])
    return m


i = int(args.count)


f =  h5py.File(args.input_file, 'r')
num = f['X_train'][:].shape[0]
number = 10**(len(str(num)) -1)
int = num/number
left = num - number*int
if i <int:
    X_train = f['X_train'][i*number:(i+1)*number,:,:]
    imgs_train = np.ndarray([number,128,128,16])
else:
    X_train = f['X_train'][i*number:,:,:]
    imgs_train = np.ndarray([left,128,128,16])
if i <int:
    X_train = f['X_train'][i*number:(i+1)*number,:,:]
    imgs_train = np.ndarray([number,128,128,16])
else:
    X_train = f['X_train'][i*number:,:,:]
    imgs_train = np.ndarray([left,128,128,16])




if i <int:
    for j in tqdm(range(number)):
        imgs_train[j]= convert_to_image(X_train[j])
else:
    for j in tqdm(range(left)):
        imgs_train[j]= convert_to_image(X_train[j])

with h5py.File('/home/chenxupeng/projects/deepshape/data/new/train_'+args.count) as t:
    t.create_dataset('train_images',data = imgs_train)
