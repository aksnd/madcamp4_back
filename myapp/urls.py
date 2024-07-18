# myapp/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ItemViewSet, simple_text;

router = DefaultRouter()
router.register(r'items', ItemViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('simple-text/<str:input_value>/', simple_text, name='simple_text'),
]