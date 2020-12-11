from django.urls import path
from resultpage import views

urlpatterns = [
    path('', views.index, name='index'),
    path('crop/', views.auto_crop, name='auto_crop'),
    path('brightness/', views.render_brightness, name='render_brightness'),
    path('prediction/', views.render_prediction, name='render_prediction'),
    path('label/', views.add_label, name='add_label'),
    path('delete/', views.delete, name='delete'),
]