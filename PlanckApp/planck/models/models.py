from django.db import models
from django.db.models.signals import post_delete
from models.cleanup import file_cleanup
import os
from django.conf import settings

class Image(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField()
    def __init__(self, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)
        self._meta.get_field('image').upload_to = "testImages"

class TrainingImage(models.Model):
    label = models.CharField(max_length=100)
    image = models.ImageField()
    def __init__(self, *args,**kwargs):
        super(TrainingImage, self).__init__(*args,**kwargs)
        self._meta.get_field('image').upload_to = os.path.join("trainingimages", self.label)

class Result(models.Model):
    instance = models.IntegerField(default=0)
    result = models.CharField(max_length=100)
    probability = models.CharField(max_length=100)
    output = models.CharField(max_length=100)
    image = models.ForeignKey(Image, on_delete=models.CASCADE)

class Weights(models.Model):
    name = models.CharField(max_length=100)

post_delete.connect(file_cleanup, sender=TrainingImage, dispatch_uid="gallery.image.file_cleanup")
