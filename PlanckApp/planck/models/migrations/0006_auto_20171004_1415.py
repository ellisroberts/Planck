# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-10-04 14:15
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('models', '0005_auto_20171004_1355'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(default='/home/ellisr/Plancknew/Planck/PlanckApp/planck/planck/planck/static/default_image/default.jpg', upload_to='testImages'),
        ),
    ]
