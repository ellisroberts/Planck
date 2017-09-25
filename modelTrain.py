from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image
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

def getResizedIm(path, height, width):
    # Load as grayscale
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    # Reduce size
    return img

def TrainModel(model):

    images = []
    labels = []
    #Grab the imagedirectoryies
    root = settings.MEDIA_ROOT
    dirlist = [ os.path.join(root,item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    
    #for each of the directories, grab all of the images
    for directory in dirlist:
        images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        for image in images[:1]: 
            #build up labels and images folder
            labels.append(directory)
            resizedImage = getResizedIm(os.path.join(directory, image), 224, 224)
            images.append(resizedImage)

    model.fit(images, labels)
    return model

    #recompile model if needed
    #run fit function to train
    #save updates model to fule
