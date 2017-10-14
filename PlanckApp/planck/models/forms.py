from django import forms
from .models import TrainingImage
from multiupload.fields import MultiImageField
from .widgets import MutuallyExclusiveRadioWidget
from .fields import  MutuallyExclusiveValueField

class ImageFileForm(forms.Form):
    image = forms.ImageField()

class MultipleImageForm(forms.Form):
    images = MultiImageField(min_num=1, max_num=30, max_file_size=1024*1024*5)
    className = forms.CharField(max_length=100)

class ModelForm(forms.Form):
    modelChoices = [("vgg19","vgg19"), ("vgg16","vgg16"), ("resnet50","resnet50"),
                    ("xception","xception"), ("inceptionV3","inceptionV3")]
    newChoices = [("mobilenet","mobilenet")]
    modelInfo = MutuallyExclusiveValueField(fields = (forms.CharField(label="cnn"), forms.CharField(label="custom")),
                                                            widget=MutuallyExclusiveRadioWidget(
                                                              labels=("Choose Existing Model", "Start with New Model"),
                                                              widgets=(forms.Select(choices=modelChoices),
                                                                          forms.Select(choices=newChoices))  
                                                             ))
