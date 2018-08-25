# -------------------------------------------------------------------------------------
# Explore training/validation X and Y data for coral
# by Austin Shen

# -------------------------------------------------------------------------------------
# packages and constants

# default packages
import numpy as np
import scipy as sc
import imageio
import matplotlib.pyplot as plt
import os.path
import sys
import random
import collections
import argparse

# paths
data_path = '../data/'
train_images_path = '../data/images_training/'
val_images_path = '../data/images_validation/'
image_path = '../plots/'

# argument parser from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required=True, help='index of image you want to view')
ap.add_argument("-b", "--box_size", required=True, help='box size dimensions')
args = vars(ap.parse_args())
index = int(args['index'])
box = int(args['box_size'])

# -------------------------------------------------------------------------------------
# read file

# open file
with open(data_path+'validation_dataset.csv') as f:
    lines = f.readlines()
print('shape of input file: %s' % np.array(lines).shape)
n = np.array(lines).shape[0]

# extract data from read file
val_Y = []
i = 0
for num,value in enumerate(lines[1:]):
    data = value.strip().split(',')
    id_value = int(data[0])
    row_num = int(data[1])
    col_num = int(data[2])
    label = str(data[3])
    label_code = str(data[4])
    fn_group = str(data[5])
    filename = str(data[6])
    method = str(data[7])
    val_Y.append([id_value, row_num, col_num, label, fn_group, filename])

# converting read data to np array (string)
val_Y = np.array(val_Y)
print('shape of output: '+str(val_Y.shape))
print('kept columns: id, row, col, label, functional_group, filename')

# -------------------------------------------------------------------------------------
# adding our own annotations

# summarise labels
labels = val_Y[:,3]
print(labels.shape)
label_counts = collections.Counter(labels)
print(label_counts)

# regrouping coral
labels_upd = []
for i in range(labels.shape[0]):
    if (labels[i] == 'Dead Substrate'):
        labels_upd.append("Bleached Coral")
    elif((str(labels[i])=='Turf')|(str(labels[i])=='Sand')|(str(labels[i])=='Turf sand')|(str(labels[i])=='Water')):
        labels_upd.append('Background')
    else:
        labels_upd.append("Healthy Coral")

# relabelling data to input new labels
labels_upd = np.array(labels_upd).reshape((len(labels_upd),1))

# -------------------------------------------------------------------------------------
# visualisation

# read image to numpy array (example)
img = imageio.imread(val_images_path+val_Y[index,5])
print('shape of image: '+str(img.shape))

# draw bounding box
print('drawing bounding box')
data_index = val_Y[index,:]
row = int(data_index[1])
col = int(data_index[2])

#box = 50
box_img = img
box_img[row-box:row+box,col-box,:] = [255,0,0]
box_img[row-box:row+box,col-box-1,:] = [255,0,0]
box_img[row-box:row+box,col+box,:] = [255,0,0]
box_img[row-box:row+box,col+box+1,:] = [255,0,0]
box_img[row-box,col-box:col+box,:] = [255,0,0]
box_img[row-box-1,col-box:col+box,:] = [255,0,0]
box_img[row+box,col-box:col+box,:] = [255,0,0]
box_img[row+box+1,col-box:col+box,:] = [255,0,0]
plt.imshow(box_img)
plt.title(str(labels[index])+' ('+str(labels_upd[index,0])+')') # title needs to be corrected
plt.show()

# -------------------------------------------------------------------------------------
