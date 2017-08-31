# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from .models import Result, Image
# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadFileForm

import pdb

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
            image = Image(name=imageFile)
            image.save()
        else:
            image = Image.objects.get(name=imageFile)
        
        #if form.is_valid():
        predictions = gradcam.returnPredictions(imageFile, weights, cnn, numResults)

        request.session['predictions'] = json.dumps(list(predictions))
        request.session['imageName'] = imageFile

        return HttpResponseRedirect(reverse("models:classify"))
    else:
         form = UploadFileForm()   
    return render(request, 'models/models.html', {'form':form})

def classify(request):
    
    #Pull prediction results to display
    predictions=request.session['predictions']
    imageName = request.session['imageName']
    
    for index in range(len(predictions)):
        result = Result(output=predictions[0], result=predicitons[1], probability=predictions[2], image=imageName)

    image = get_object_or_404(Image, name=imageName)
    return render(request, 'models/classify.html', {'image':image})

def train(request):
    return HttpResponse('This Page will be used to update existing models');
