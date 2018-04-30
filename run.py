import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import random
import numpy as np
from tqdm import tqdm
import cv2
from scipy import ndimage, misc
from scipy.misc import imread, imresize
from keras import regularizers
from keras.applications.xception import *
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from skimage import feature, exposure, color
from skimage.morphology import disk
from skimage.filters import rank
from skimage.transform import resize
import glob
import pandas as pd
import tensorflow as tf

# Knobs & Levers
# This is the name of the neural net I'm using
network_name = '12_Dense'
# These can be adjusted depending on which preprocessing algorithm(s) you would like to use.
processing_type = 'CIELUV_Gen'
# Use the pre-segmented images with Gabor Vecsei's thresholding algorithm?
using_vecsei_segs = False
# Use the Histogram Equalizer?
using_eq = False
# If using Histogram equalizer, use HSV equalization?
using_hsv = False
# Segment using CIELUV thresholding?
using_cieluv = True
# Use an ImageDataGenerator with rotation, stretching, zooming, and flipping?
using_generator = True
# Display ten sample images from the dataset?
display_samples = False
# How many epochs?
epoch = 10
# What's your desired batch size?
batch = 50
# What size image? (images are size x size x 3)
size = 300

# Parameters for Cieluv
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('dark_background')

def cieluv(img, target):
    # adapted from https://www.compuphase.com/cmetric.htm
    img = img.astype('int')
    aR, aG, aB = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    bR, bG, bB = target
    rmean = ((aR + bR) / 2.).astype('int')
    r2 = np.square(aR - bR)
    g2 = np.square(aG - bG)
    b2 = np.square(aB - bB)

    # final sqrt removed for speed; please square your thresholds accordingly
    result = (((512 + rmean) * r2) >> 8) + 4 * g2 + (((767 - rmean) * b2) >> 8)

    return result


# Equalization Algorithm
def equalize(img):
    rescale = img
    if using_hsv:
        rescale = color.rgb2hsv(rescale)
    rescale[:, :, 0] = exposure.equalize_hist(rescale[:, :, 0])
    rescale[:, :, 1] = exposure.equalize_hist(rescale[:, :, 1])
    rescale[:, :, 2] = exposure.equalize_hist(rescale[:, :, 2])
    if using_hsv:
        rescale = color.hsv2rgb(rescale)
    return rescale


# All possible image labels for this set
labels = ['Black-grass',
          'Charlock',
          'Cleavers',
          'Common Chickweed',
          'Common wheat',
          'Fat Hen',
          'Loose Silky-bent',
          'Maize',
          'Scentless Mayweed',
          'Shepherds Purse',
          'Small-flowered Cranesbill',
          'Sugar beet']


images = []
images_y = []
test_images = []
test_images_y = []
isize = (size, size, 3)

# Using raw images or Vecsei segmented images?
main_path = ''
if using_vecsei_segs:
    main_path = "C:\\Users\\Tolar\\Desktop\\SEEDLING\\seg_train\\"
else:
    main_path = "C:\\Users\\Tolar\\Desktop\\SEEDLING\\train\\"

# Load all images from the selected index
i = 0
for index, label in tqdm(enumerate(labels), total=len(labels)):
    for file in os.listdir(main_path + label):
        image = imread((main_path + '/{}/{}').format(label, file))
        image = imresize(image[:, :, :3], (size, size))
        pick = random.random()
        if pick < 0.9:
            images.append(np.asarray([image]))
            images_y.append(index)
        else:
            test_images.append(np.asarray([image]))
            test_images_y.append(index)
        i += 1

images_y = np_utils.to_categorical(images_y, 12)
test_images_y = np_utils.to_categorical(test_images_y, 12)

images = np.vstack(images)
images_y = np.vstack(images_y)

test_images = np.vstack(test_images)
test_images_y = np.vstack(test_images_y)

print ("Image loading: Complete")

# Process Training Data

# Histogram equalization
if using_eq:
    for i in range(len(images)):
        im = images[i]
        rescale = equalize(im)
        images[i] = rescale

# CIELUV segmentation
if using_cieluv:
    for i in range(len(images)):
        img = images[i]
        img_filter = (
            (cieluv(img, (71, 86, 38)) > 1600)
            & (cieluv(img, (65,  79,  19)) > 1600)
            & (cieluv(img, (95,  106,  56)) > 1600)
            & (cieluv(img, (56,  63,  43)) > 500)
        )
        img[img_filter] = 0
        img = cv2.medianBlur(img, 9)
        images[i] = img

print ("Training data processed")

# Process Test Data

# Histogram equalization
if using_eq:
    for i in range(len(test_images)):
        imy = test_images[i]
        rescale = equalize(imy)
        test_images[i] = rescale

# CIELUV segmentation
if using_cieluv:
    for i in range(len(test_images)):
        img = test_images[i]
        img_filter = (
            (cieluv(img, (71, 86, 38)) > 1600)
            & (cieluv(img, (65, 79, 19)) > 1600)
            & (cieluv(img, (95, 106, 56)) > 1600)
            & (cieluv(img, (56, 63, 43)) > 500)
        )
        img[img_filter] = 0
        img = cv2.medianBlur(img, 9)
        test_images[i] = img

# Displays the first ten items in the test data
if display_samples:
    for i in range(10):
        img = test_images[i]
        if i < 10:
            plt.imshow(img)
            plt.show()

print ("test data processed")

images = images.astype('float32')
test_images = test_images.astype('float32')
images /= 255
test_images /= 255

# An image data generator set to rotate, stretch, zoom, and flip images
if using_generator:
    generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                      rotation_range=180,
                                      width_shift_range=0.3,
                                      height_shift_range=0.3,
                                      zoom_range=0.3,
                                      horizontal_flip=True,
                                      vertical_flip=True)

    generator.fit(images)

if 1:
    # Untrainable Xception layer
    base = Xception(weights='imagenet', input_shape=isize, pooling='max', include_top=False)
    for layer in base.layers:
        layer.trainable = False
    pre_model = base.output
    dense_layers = []
    for i in range(len(labels)):
        item = Dense(512, activation='tanh')(pre_model)
        item = Dropout(0.25)(item)
        item = BatchNormalization()(item)
        item = Dense(1, activation='sigmoid')(item)
        dense_layers.append(item)
    dense_out = concatenate(dense_layers)
    output_layer = Dense(12, activation='softmax')(dense_out)

    model = Model(inputs=base.input, outputs=output_layer)

    # Model uses Adadelta optimizer for quick regression to an ideal point
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    if using_generator:
        model.fit_generator(generator.flow(images, images_y, batch_size=batch), epochs=epoch, verbose=1)
    else:
        model.fit(images, images_y, batch_size= batch, epochs=epoch, verbose=1)
    test_images_y = test_images_y.argmax(axis= -1)
    prob = model.predict(test_images, batch_size=batch, verbose=1)
    predictions = prob.argmax(axis= -1)
    count = np.size(test_images_y, 0)
    correct = 0
    for i in range(count):
        if test_images_y[i] == predictions[i]:
            correct = correct + 1
    accuracy = correct/count
    print(str(processing_type) + " Score on " + network_name + ":")
    print("Accuracy: " + str(accuracy * 100) + "%")