from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from django.conf import settings
import os
import pdb
import math
import json

def getResizedIm(path, height, width):
    # Load as grayscale
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    # Reduce size
    return img

def TrainModel(model):

    trainData = []
    labels = []
    index = 0
    #Grab the imagedirectoryies
    root = settings.MEDIA_ROOT
    imageDir = os.path.join(root, "trainingimages")
    classes = json.load(open(os.path.join(root, "class_index.json")))    
    num_classes = len(classes)

    #get the dimensiosn of the input
    shape = model.get_input_shape_at(0)
    height = shape[1]
    width = shape[2]
    depth = shape[3]

    #go through dict and build up the label and training arrays
    for classNum in classes:
        directory = classes[classNum][1]
        curDir = os.path.join(imageDir,directory)
        images = [image for image in os.listdir(os.path.join(imageDir,directory)) \
                 if os.path.isfile(os.path.join(curDir, image))]
        for image in images:
            img = cv2.imread(os.path.join(curDir, image))
            resized = cv2.resize(img, (224,224))
            trainData.append(resized)
            labels.append(classNum)
 
    trainData = np.array(trainData, dtype=np.uint8)
    trainData = trainData.reshape(trainData.shape[0], height, width, depth)
    trainData = trainData.astype('float32')
    trainData /= 255

    labels = np.array(labels, dtype=np.uint8)
    labels = to_categorical(labels, num_classes=1000)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(trainData, labels, batch_size=1, epochs = 1)
    return model
