from django.shortcuts import render, get_object_or_404
from .models import Result, Image
# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
from .forms import *
from django.core.urlresolvers import reverse

from keras.models import load_model
import pdb
import json
import numpy as np
import time

import sys
sys.path.append("../..")
import gradcam
import modelTrain 

def index(request):
    preTrainedModelForm = None
    emptyForm = None

    if request.method == 'POST':
        #Get the paramters to use for prediction
        imageFile = request.FILES['image']
        numResults = request.POST['numresults']

        if (request.POST['modelSelect'] == 'trained'):
            preTrainedModelForm = PreTrainedModelForm(request.POST, request.FILES)
            #if (preTrainedModelForm.is_valid()):
            ##TODO: Throw exception and message prompting user to upload proper file
            model = loadModel(request.FILES['modelFile'])

        else:
            if (request.POST['modelSelect'] == 'nottrained'):
               ##TODO Verify that the form fields are valid, if not, clear the form fields and rerender (optimal way?)
               data = {'cnn': request.POST['cnn'], 'weights': request.POST['weights']}
               emptyForm = DefaultModelForm(initial=data)
               #if (emptyForm.is_valid()):
               model = gradcam.instantiateModel(request.POST['cnn'], request.POST['weights'])

        ##TODO: Generate a prompt here to save the file

        #TODO: handle exception and prompt user to upload proper image file
        predictions = gradcam.returnPredictions(imageFile, model, int(numResults))

        model.save('current.hdf5')
        #create Image object
        if (len(list(Image.objects.all())) == 0):
            image = Image(name=imageFile.name)
            image.save()
        else:
            image = Image.objects.get(name=imageFile.name)
            Result.objects.all().delete()

        #need to properly serialize predictions to output them on webpage
        request.session['predictions'] = json.dumps((np.array(predictions)).tolist())
        request.session['imageName'] = imageFile.name

        return HttpResponseRedirect(reverse("models:classify"))
    else:
        preTrainedModelForm = PreTrainedModelForm()
        defaultModelForm = DefaultModelForm()
        imageFileForm = ImageFileForm()
    return render(request, 'models/models.html', {'preTrained':preTrainedModelForm, 'default':defaultModelForm, 'image':
 ImageFileForm})

def classify(request):

    if (request.method == "POST"):
        return HttpResponseRedirect(reverse("Models:train"))

    else:
        #Create json Decoder to convert json object back into python list
        jsonDec = json.decoder.JSONDecoder()

        #Pull prediction results to display
        predictions=jsonDec.decode(request.session['predictions'])
        imageName = request.session['imageName']

        curImage = get_object_or_404(Image, name=imageName)
        for index in range(len(predictions)):
           result = Result(instance=index, output=predictions[index][0], result=predictions[index][1], probability=predictions[index][2], image=curImage)
           result.save()
           curImage.result_set.add(result)

    return render(request, 'models/classify.html', {'image':curImage})

def train(request):
    if (request.method == "POST"):
        #model = load_model(request.session['current'])
        model = load_model('current.hdf5')
        model = modelTrain.TrainModel(model)
        return HttpResponseRedirect(reverse("Models:Index"))
    return render(request, 'models/train.html')
