from django import forms

class PreTrainedModelForm(forms.Form):
    modelFile = forms.FileField()

class DefaultModelForm(forms.Form):
    cnn = forms.CharField()
    weights = forms.CharField()

class ImageFileForm(forms.Form):
    image = forms.ImageField()
    def __init__(self, isMultiple=False):
        super(ImageFileForm, self).__init__()
        self.fields['image'] = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': isMultiple}))
