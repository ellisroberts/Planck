from django.shortcuts import render, get_object_or_404
from .models import Result, Image
# Create your views here.

from django.forms import formset_factory
from django.forms.models import modelformset_factory
from django.http import HttpResponse, HttpResponseRedirect
from .forms import *
from django.core.urlresolvers import reverse
from django.contrib import messages
from django.conf import settings

import keras.applications
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import pdb
import json
import numpy as np
import time

import os
import sys
sys.path.append("../..")
import gradcam
import modelTrain 

def index(request):

    modelForm = None
    if request.method == 'POST':
        #Get the paramters to use for prediction
        imageFile = request.FILES['image']
        numResults = request.POST['numresults']
 
        #Did we start from a fresh cnn
        request.session['freshModel'] = True

        #TODO validate the form
        if (request.POST['modelInfo_radio'] == '0'):
             model = gradcam.instantiateModel('imagenet', request.POST['cnn'])
             request.session['freshModel'] = False
        else:
             model = gradcam.instantiateModel('None')

        #TODO: handle exception and prompt user to upload proper image file
        predictions = gradcam.returnPredictionsandGenerateHeatMap(imageFile, model, int(numResults))

        model.save('current.hdf5')

        #create Image object
        try:
            image = Image.objects.get(name=imageFile.name)
        except Image.DoesNotExist:
            image = Image(image=imageFile, name=imageFile.name)
            image.save()

        #Want to restart with fresh results
        Result.objects.all().delete()

        #need to properly serialize predictions to output them on webpage
        request.session['predictions'] = json.dumps((np.array(predictions)).tolist())
        request.session['imageName'] = imageFile.name
        request.session['numresults'] = numResults
  
        return HttpResponseRedirect(reverse("models:classify")) 
    else:
        modelForm = ModelForm()
        imageFileForm = ImageFileForm() 
    return render(request, 'models/models.html', {'model':modelForm, 'image':imageFileForm})

def classify(request):
    if (request.method == "POST"):
        if (request.POST["reTrain"] == "Yes"):
            #delete all images added by user
            TrainingImage.objects.all().delete()
            Image.objects.all().delete()
            return HttpResponseRedirect(reverse("models:index"))

        if (request.POST["reTrain"] == "No"):
            return HttpResponseRedirect(reverse("models:train"))

    else:
        #Create json Decoder to convert json object back` into python list
        jsonDec = json.decoder.JSONDecoder()

        #Pull prediction results to display
        predictions=jsonDec.decode(request.session['predictions'])
        imageName = request.session['imageName']

        #Get the Results for the Image
        curImage = get_object_or_404(Image, name=imageName)
        for index in range(len(predictions)):
           result = Result(instance=index, output=predictions[index][0], result=predictions[index][1], probability=predictions[index][2], image=curImage)
           result.save()
           curImage.result_set.add(result)

    #If we are starting with a fresh model, give user the option to retrain
    if (request.session['freshModel']):        
        return render(request, 'models/classifyredirect.html', {'image':curImage})
    
    return render(request, 'models/classify.html', {'image':curImage})

def train(request):
    testImagePath = os.path.join(settings.MEDIA_ROOT,"testImages")
    if (request.method == "POST"):
    
        #upload set of images to add more training data for requested class    
        form = MultipleImageForm(request.POST, request.FILES)
        for image in request.FILES.getlist('images'):
            trainImage = TrainingImage(image=image, label=request.POST['className'])
            trainImage.save()

        #load the model, train it, and save it again
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            model = load_model('current.hdf5')
        model = modelTrain.TrainModel(model)
        model.save('current.hdf5')

        #run inference on updated model
        imageName = request.session['imageName']
        imagePath = os.path.join(testImagePath, imageName)

        numResults = request.session['numresults']        

        predictions = gradcam.returnPredictionsandGenerateHeatMap(imagePath, model, int(numResults))
        request.session['predictions'] = json.dumps((np.array(predictions)).tolist())

        return HttpResponseRedirect(reverse("models:classify"))
    else:
        imageUploader = MultipleImageForm()
    return render(request, 'models/train.html', {"imageUploader":imageUploader}
)
