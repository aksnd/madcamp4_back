import yfinance as yf
from datetime import datetime, timedelta

def get_stock_price_on_date(ticker, date_str):
    """
    주어진 날짜를 기준으로 전날, 해당 날짜, 다음날의 주가를 반환합니다.
    
    Parameters:
    - ticker: 주식 티커
    - date_str: 날짜 문자열 (예: '2024-07-18')
    
    Returns:
    - prices: dict {'previous_day': float, 'given_day': float, 'next_day': float}
    - error: str
    """
    try:
        # 입력된 날짜 문자열을 날짜 객체로 변환
        date = datetime.strptime(date_str, '%Y-%m-%d')
        previous_day = date - timedelta(days=1)
        next_day = date + timedelta(days=1)
        
        # 티커 데이터를 가져옴
        stock = yf.Ticker(ticker)
        history = stock.history(start=previous_day.strftime('%Y-%m-%d'), end=(next_day + timedelta(days=1)).strftime('%Y-%m-%d'))
        
        if history.empty:
            return None, "Ticker data not available"

        # 주가 데이터를 추출
        prices = {
            'previous_day': history.loc[previous_day.strftime('%Y-%m-%d')]['Close'] if previous_day.strftime('%Y-%m-%d') in history.index else None,
            'given_day': history.loc[date.strftime('%Y-%m-%d')]['Close'] if date.strftime('%Y-%m-%d') in history.index else None,
            'next_day': history.loc[next_day.strftime('%Y-%m-%d')]['Close'] if next_day.strftime('%Y-%m-%d') in history.index else None
        } #영업일 기준으로 7일을 가져와서 리스트로 관리하기 
        return prices, None
    except Exception as e:
        return None, str(e)
