from django.urls import re_path

from . import views

urlpatterns = [
    # /api/1/43/
    re_path(r'^(?P<id>\d+)/(?P<level>.+)/$', views.add_entry, name='add_entry'),
]
