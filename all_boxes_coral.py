# -------------------------------------------------------------------------------------
# View all bounding boxes in images of choice
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
ap.add_argument("-i", "--index", required=True, help='index of image in validation set (0-1042)')
ap.add_argument("-b", "--box_size", required=True, help='box size dimensions')
args = vars(ap.parse_args())
index_choice = int(args['index'])
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

# number of unique images
unique_images = list(set(val_Y[:,5]))
print('unique images: %s' % len(unique_images))

# writing script to draw red bounding boxes on image from input
def boxes(img, row, col, box, label):
    if (label == 'Bleached Coral'):
        img[row-box:row+box,col-box,:] = [255,0,0]
        img[row-box:row+box,col-box-1,:] = [255,0,0]
        img[row-box:row+box,col+box,:] = [255,0,0]
        img[row-box:row+box,col+box+1,:] = [255,0,0]
        img[row-box,col-box:col+box,:] = [255,0,0]
        img[row-box-1,col-box:col+box,:] = [255,0,0]
        img[row+box,col-box:col+box,:] = [255,0,0]
        img[row+box+1,col-box:col+box,:] = [255,0,0]
    elif(label == 'Healthy Coral'):
        img[row-box:row+box,col-box,:] = [0,255,0]
        img[row-box:row+box,col-box-1,:] = [0,255,0]
        img[row-box:row+box,col+box,:] = [0,255,0]
        img[row-box:row+box,col+box+1,:] = [0,255,0]
        img[row-box,col-box:col+box,:] = [0,255,0]
        img[row-box-1,col-box:col+box,:] = [0,255,0]
        img[row+box,col-box:col+box,:] = [0,255,0]
        img[row+box+1,col-box:col+box,:] = [0,255,0]
    return(img)

# show all boxes
for i in range(index_choice,index_choice+1):
    filename = unique_images[i]
    print('image: '+str(filename))
    img = imageio.imread(val_images_path+filename)
    indices = [i for i,x in enumerate(val_Y[:,5]) if x == filename]
    for j in range(0,len(indices)):
        index = indices[j]
        row = int(val_Y[index,1])
        col = int(val_Y[index,2])
        img = boxes(img, row, col, 50, labels_upd[index])
    plt.imshow(img)
    plt.title(filename)
    plt.show()

# -------------------------------------------------------------------------------------
