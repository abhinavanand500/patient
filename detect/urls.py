from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('gethtml', views.gethtml, name='gethtml'),
    path('about', views.about, name='about'),
    path('contact', views.contact, name='contact'),
    path('newHtml', views.newHtml, name='newHtml'),
]
