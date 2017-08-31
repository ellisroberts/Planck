from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=100)

class Result(models.Model):
    result = models.CharField(max_length=100)
    probability = models.FloatField()
    output = models.IntegerField()
    image = models.ForeignKey(Image, on_delete=models.CASCADE)

