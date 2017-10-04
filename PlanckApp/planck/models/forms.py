from django import forms
from .models import TrainingImage
from multiupload.fields import MultiImageField

class PreTrainedModelForm(forms.Form):
    modelFile = forms.FileField()

class DefaultModelForm(forms.Form):
    cnn = forms.CharField()
    weights = forms.CharField()

class ImageFileForm(forms.Form):
    image = forms.ImageField()

class MultipleImageForm(forms.Form):
    images = MultiImageField(min_num=1, max_num=30, max_file_size=1024*1024*5)
    label = forms.CharField(max_length=100)

