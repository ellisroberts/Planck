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
    root = os.path.join(settings.MEDIA_ROOT, "trainingimages")

    dirList = [directory for directory in os.listdir(root) if os.path.isdir(os.path.join(root,directory))]
    for directory in dirList:
        curDir = os.path.join(root,directory)
        images = [image for image in os.listdir(os.path.join(root,directory)) \
                 if os.path.isfile(os.path.join(curDir, image))]
        for image in images:
            img = cv2.imread(os.path.join(curDir, image))
            resized = cv2.resize(img, (224,224))
            trainData.append(resized)
            labels.append(index)
        index += 1
 
    trainData = np.array(trainData, dtype=np.uint8)
    trainData = trainData.reshape(trainData.shape[0], 224, 224, 3)

    labels = np.array(labels, dtype=np.uint8)
    labels = to_categorical(labels, num_classes=len(labels))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(trainData, labels, epochs = 1)
    return model
