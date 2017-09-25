from django import forms

class PreTrainedModelForm(forms.Form):
    modelFile = forms.FileField()

class DefaultModelForm(forms.Form):
    cnn = forms.CharField()
    weights = forms.CharField()

class ImageFileForm(forms.Form):
    image = forms.ImageField()
