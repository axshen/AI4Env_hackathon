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
write_dir = '../data/classification/val/'

# argument parser from command line
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--box_size", required=True, help='box size dimensions')
args = vars(ap.parse_args())
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
# save images to relevant file

print('writing images to directory')

# writing image for each directory
healthy_coral_count = 0
bleached_coral_count = 0
background_count = 0
for i in range(0,n):
    try:
        print('working on img '+str(i))
        img = imageio.imread(val_images_path+val_Y[i,5])
        row = int(val_Y[i,1])
        col = int(val_Y[i,2])
        subset_img = img[row-box:row+box, col-box:col+box, :]
        if (labels_upd[i] == 'Bleached Coral'):
            imageio.imwrite(write_dir+'bleached_coral/'+str(bleached_coral_count)+'.png', subset_img)
            bleached_coral_count += 1
        elif (labels_upd[i] == 'Healthy Coral'):
            imageio.imwrite(write_dir+'healthy_coral/'+str(healthy_coral_count)+'.png', subset_img)
            healthy_coral_count += 1
        else:
            imageio.imwrite(write_dir+'background/'+str(background_count)+'.png', subset_img)
            background_count += 1
    except:
        pass

print('file saving completed')

# -------------------------------------------------------------------------------------
