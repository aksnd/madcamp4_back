# myapp/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ItemViewSet, simple_text, PredictView,CheckPrice, chatbot_response, EmotionView, recommend_company, kakao_login,kakao_callback;

router = DefaultRouter()
router.register(r'items', ItemViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('simple-text/<str:input_value>/', simple_text, name='simple_text'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('emotion/',EmotionView.as_view(),name='emotion'),
    path('check_price/',CheckPrice.as_view(),name='predict'), #http://52.78.53.98:8000/check_price/?ticker=005930.KS&date=2024-07-18 이런식으로 사용가능
    path('api/chatbot/', chatbot_response, name='chatbot_response'),
    path('api/recommend/', recommend_company, name='recommend_company'),
    path('kakao/login/', kakao_login, name='kakao_login'),
    path('kakao/callback/', kakao_callback, name='kakao_callback'),
]