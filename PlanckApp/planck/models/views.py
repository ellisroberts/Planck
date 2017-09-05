# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, get_object_or_404
from .models import Result, Image
# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadFileForm
from django.core.urlresolvers import reverse

import json
import numpy as np

import sys
sys.path.append("../..")
import gradcam

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        #Get the paramters to use for prediction
        imageFile = request.FILES['file']
        weights = request.POST['weights']
        cnn = request.POST['cnn']
        numResults = request.POST['numResults']

        #create Imagw object
        if (len(list(Image.objects.all())) == 0):
            image = Image(name=imageFile.name)
            image.save()
        else:
            image = Image.objects.get(name=imageFile.name)
            Result.objects.all().delete()
        
        #if form.is_valid():
        predictions = gradcam.returnPredictions(imageFile, weights, cnn, int(numResults))

        #need to properly serialize predictions to output them on webpage
        request.session['predictions'] = json.dumps((np.array(predictions)).tolist())
        request.session['imageName'] = imageFile.name

        return HttpResponseRedirect(reverse("models:classify"))
    else:
         form = UploadFileForm()   
    return render(request, 'models/models.html', {'form':form})

def classify(request):
    
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
    return HttpResponse('This Page will be used to update existing models');
