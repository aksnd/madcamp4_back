import yfinance as yf
from datetime import datetime, timedelta

def change_company_to_ticker(company):
    company_to_ticker = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "LG에너지솔루션": "373220.KS",
        "삼성바이오로직스": "207940.KS",
        "현대차": "005380.KS",
        "기아": "000270.KS",
        "셀트리온": "068270.KS",
        "KB금융": "105560.KS",
        "POSCO홀딩스": "005490.KS",
        "NAVER": "035420.KS",
        "삼성생명": "032830.KS",
        "LG전자": "066570.KS",
        "한화오션": "042660.KS",
        "고려아연": "010130.KS",
        "금양": "001570.KS",
        "현대해상": "001450.KS"
    }
    
    return company_to_ticker.get(company, None)


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
    
def get_last_business_day(target_date):
    """
    주어진 날짜가 영업일인지 확인하고, 그렇지 않다면 가장 가까운 전 영업일을 반환합니다.
    :param target_date: 확인할 날짜 (datetime.date 객체)
    :return: 영업일 (datetime.date 객체)
    """
    ticker = yf.Ticker("AAPL")  # 어떤 회사든 상관없음, 단순히 영업일을 확인하기 위한 용도
    start_date = target_date - timedelta(days=30)
    end_date = target_date + timedelta(days=1)
    hist = ticker.history(start=start_date, end=end_date).index  # target_date 이전 30일간의 데이터 가져오기
    hist_dates = [date.date() for date in hist]  # 인덱스의 날짜 부분 추출
    if target_date in hist_dates:
        return target_date
    else:
        # 주어진 날짜가 영업일이 아니면 가장 가까운 전 영업일을 찾음
        previous_dates = [date for date in hist_dates if date < target_date]
        if not previous_dates:
            raise ValueError("No previous business days found in the last 30 days.")
        last_business_day = max(previous_dates)
        return last_business_day


def get_stock_price(company_name, date):
    """
    주어진 회사 티커와 날짜에 대해 주가 또는 가장 가까운 전 영업일의 종가를 반환합니다.
    :param company_ticker: 회사 티커
    :param date: 확인할 날짜 (datetime.date 객체)
    :return: 주가 또는 None
    """
    company_ticker =change_company_to_ticker(company_name)
    if(company_ticker == None):
        return None
    try:
        last_business_day = get_last_business_day(date)
    except ValueError as e:
        print(e)
        return None

    ticker = yf.Ticker(company_ticker)
    hist = ticker.history(start=last_business_day, end=last_business_day + timedelta(days=1))
    if not hist.empty:
        return hist['Close'].iloc[0]
    else:
        return None


def get_stock_ratio(company_name, date):
    """
    주어진 날짜의 주가와 그 다음 영업일의 주가를 나눈 값을 반환합니다.
    :param company_ticker: 회사 티커
    :param date: 확인할 날짜 (datetime.date 객체)
    :return: 주가 비율 또는 None
    """
    company_ticker =change_company_to_ticker(company_name)
    if(company_ticker == None):
        return None
    last_business_day = get_last_business_day(date)
    
    # 해당 날짜로부터 1주일 간의 주가 데이터를 가져옴
    ticker = yf.Ticker(company_ticker)
    one_week_data = ticker.history(start=last_business_day, end=last_business_day + timedelta(days=7))
    
    if len(one_week_data) < 2:
        return 0.0

    price_today = one_week_data['Close'].iloc[0]
    price_next_day = one_week_data['Close'].iloc[1]

    if price_today is None or price_next_day is None:
        return 0.0

    return (price_next_day / price_today)-1.0


def get_historical_stock_data(company, start_date, end_date):
    company_ticker =change_company_to_ticker(company)
    stock = yf.Ticker(company_ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist.reset_index().to_dict(orient='records')