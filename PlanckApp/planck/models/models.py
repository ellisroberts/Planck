from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=100)

class TrainingImage(models.Model):
    label = models.CharField(max_length=100)
    image = models.ImageField()

    def __init__(self, label):
        self.label = label
        self.image = models.ImageField(upload_to=label)

class Result(models.Model):
    instance = models.IntegerField(default=0)
    result = models.CharField(max_length=100)
    probability = models.CharField(max_length=100)
    output = models.CharField(max_length=100)
    image = models.ForeignKey(Image, on_delete=models.CASCADE)

class Weights(models.Model):
    name = models.CharField(max_length=100)
