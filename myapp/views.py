from rest_framework import viewsets
from .models import Item
from .serializers import ItemSerializer
from django.http import JsonResponse
from django.views import View
from .yfinance_control import get_stock_price_on_date
import yfinance as yf

class PredictView(View):
    def get(self, request):
        ticker = request.GET.get('ticker', None)
        if not ticker:
            return JsonResponse({'error': 'Ticker is required'}, status=400)
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            return JsonResponse({'prediction': current_price}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

class CheckPrice(View):
    def get(self, request):
        ticker = request.GET.get('ticker', None)
        date_str = request.GET.get('date', None)
        
        if not ticker or not date_str:
            return JsonResponse({'error': 'Ticker and date are required'}, status=400)

        prices, error = get_stock_price_on_date(ticker, date_str)
        if error:
            return JsonResponse({'error': error}, status=500)
        
        response = JsonResponse({'prices': prices}, status=200)
        return response
        
class ItemViewSet(viewsets.ModelViewSet):
    queryset = Item.objects.all()
    serializer_class = ItemSerializer
    
def simple_text(request, input_value):
    return JsonResponse({'message': f'you sent {input_value}'})