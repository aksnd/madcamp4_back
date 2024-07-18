from rest_framework import viewsets
from .models import Item
from .serializers import ItemSerializer
from django.http import JsonResponse

class ItemViewSet(viewsets.ModelViewSet):
    queryset = Item.objects.all()
    serializer_class = ItemSerializer
    
def simple_text(request, input_value):
    return JsonResponse({'message': f'you sent {input_value}'})