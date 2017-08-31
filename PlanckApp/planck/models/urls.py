from django.conf.urls import url

from . import views

app_name='models'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    #results page
    url(r'^classify', views.classify, name='classify'),
    #page for updating the model
    url(r'^train',    views.train,   name='train'),
]
