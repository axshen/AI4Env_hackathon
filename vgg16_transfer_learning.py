# ------------------------------------------------------------------------
# Script to reshape images from file to chosen dimensions
# by Austin Shen

# ------------------------------------------------------------------------
# libraries

import tensorflow as tf
from keras.utils.data_utils import get_file
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------------
# hyperparameter selection

# input image dimensions
width = 100
height = 100
batch_size = 128
epochs = 20

# naming directories
train_path = '../data/classification/train/'
validation_path = '../data/classification/val/'
test_path = '../data/classification/test/'
plot_path = '../plots/'

# ------------------------------------------------------------------------
# image generation with augmentation

datagen = image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

train_generator = datagen.flow_from_directory(
        validation_path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
		test_path,
		target_size = (width, height),
		batch_size = batch_size,
		class_mode = 'categorical')

# ------------------------------------------------------------------------
# build model

# taking VGG16 model layers
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(width, height, 3))

# adding our own model to VGG16
x = vgg_conv.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(3, activation='softmax')(x)

# creating model
model = models.Model(inputs=vgg_conv.input, outputs=predictions)
for layer in vgg_conv.layers:
    layer.trainable = False

# compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
				metrics=['accuracy'])
model.summary()

# ------------------------------------------------------------------------
# train

# train model on generated data
history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator)

# save model to loadable file
#model.save("../models/coral_classification_val.h5")

# ------------------------------------------------------------------------
# summary

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=2.0)
plt.plot(history.history['val_loss'],'b',linewidth=2.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
#plt.savefig(plot_path+'loss_curve.png')
plt.close()

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=2.0)
plt.plot(history.history['val_acc'],'b',linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
#plt.savefig(plot_path+'accuracy_curve.png')
plt.close()

# ------------------------------------------------------------------------
