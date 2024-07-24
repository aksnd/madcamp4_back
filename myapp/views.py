from rest_framework import viewsets
from .models import Item
from .serializers import ItemSerializer
from django.http import JsonResponse
from django.views import View
from bs4 import BeautifulSoup
from .yfinance_control import get_stock_price_on_date, get_stock_price, get_historical_stock_data
from .crawling import get_company_news_today, get_input_news_today
from .get_emotion import calculate_emotion_score, calculate_relevance_score
import yfinance as yf
from rest_framework.decorators import api_view
from rest_framework.response import Response
import openai
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever
from datetime import datetime, timedelta, date
import os
import numpy as np
from newspaper import Article

import requests
from django.shortcuts import redirect, render
from django.contrib.auth import login
from django.contrib.auth import logout as django_logout
#from django.contrib.auth.models import User
from .models import User  # 커스텀 User 모델 임포트

def kakao_login(request):
    app_rest_api_key = '82c56c5c2137fe9743b658368f339567'
    redirect_uri = 'http://52.78.53.98:8000/kakao/callback/'
    kakao_oauth_url = f"https://kauth.kakao.com/oauth/authorize?client_id={app_rest_api_key}&redirect_uri={redirect_uri}&response_type=code"
    
    return redirect(kakao_oauth_url)

def kakao_callback(request):
    app_rest_api_key = '82c56c5c2137fe9743b658368f339567'
    redirect_uri = 'http://52.78.53.98:8000/kakao/callback/'
    code = request.GET.get('code')
    print("kakao_callback_check")
    token_url = f"https://kauth.kakao.com/oauth/token?grant_type=authorization_code&client_id={app_rest_api_key}&redirect_uri={redirect_uri}&code={code}"
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    token_response = requests.post(token_url, headers=token_headers)
    token_json = token_response.json()
    access_token = token_json.get('access_token')

    user_url = "https://kapi.kakao.com/v2/user/me"
    user_headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
    }
    user_info_response = requests.get(user_url, headers=user_headers)
    user_info_json = user_info_response.json()
    kakao_id = user_info_json.get('id')
    
    
    user, created = User.objects.get_or_create(kakao_id=kakao_id)
    print(created)
    if created:
        user.save()
    return redirect(f'http://localhost:3002/login-success?kakao_id={kakao_id}')

def kakao_logout(request):
    access_token = request.session.get('access_token')
    if access_token:
        kakao_logout_url = 'https://kapi.kakao.com/v1/user/logout'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.post(kakao_logout_url, headers=headers)
        if response.status_code == 200:
            django_logout(request)
            return redirect('http://localhost:3002/')
    return redirect('http://localhost:3002/')

class PredictView(View):
    def get(self, request):
        company = request.GET.get('company', None)
        if not company:
            return JsonResponse({'error': 'company is required'}, status=400)
        dates = date.today()
        start_date = dates - timedelta(days=30)  # 과거 30일 간의 데이터를 가져오기 위한 시작 날짜
        end_date = dates
        try:
            price = get_stock_price(company,dates)
            articles = get_company_news_today(company,dates)
            historical_data = get_historical_stock_data(company, start_date, end_date)

            result_articles = []
            for article in articles:
                if(article['content']!=''):
                    result_articles.append({
                    'summary': article['summary'],
                    'emotion': calculate_emotion_score(article['content']),
                    'relevance': calculate_relevance_score(article['content'],company),
                    'link': article['link']
                    })
                else:
                    print("cannot get contents") #이제 발생해서는 안됨.
                    result_articles.append({
                        'summary': article['summary'],
                        'emotion': calculate_emotion_score(article['title']),
                        'relevance': calculate_relevance_score(article['title'],company),
                        'link': article['link']
                    })

            response_data =  {
                'price': price,
                'articles': result_articles,
                'historical_data': historical_data,
            }
            return JsonResponse(response_data, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
class EmotionView(View):
    def get(self, request):
        company = request.GET.get('company', None)
        if not company:
            return JsonResponse({'error': 'company is required'}, status=400)
        dates = date.today()
        try:
            articles = get_input_news_today(company,dates)
            result_articles = []
            for article in articles:
                if(article['content']!=''):
                    result_articles.append({
                    'summary': article['title'],
                    'emotion': calculate_emotion_score(article['content']),
                    'relevance': calculate_relevance_score(article['content'],company),
                    'link': article['link']
                    })
                else:
                    print("cannot get contents") 
                    result_articles.append({
                        'summary': article['title'],
                        'emotion': calculate_emotion_score(article['title']),
                        'relevance': calculate_relevance_score(article['title'],company),
                        'link': article['link']
                    })

            response_data =  {
                'articles': result_articles
            }
            return JsonResponse(response_data, status=200)
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
#openai api key
api_key = os.environ.get('OPENAI_API_KEY')
#임베딩 설정
embedding_function = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
# LLM 설정
llm = ChatOpenAI(api_key=api_key)

@api_view(['POST'])
def chatbot_response(request):
    user_input = request.data.get('input', '')
    kakao_id = request.data.get('kakaoId', '')
    news_url = request.data.get('newsUrl', '')

    if news_url:
        # 뉴스 요약 및 관련 기사 검색
        news_summary = summarize_news_link(news_url)
        save_user_question(news_summary, kakao_id)
        related_articles = search_related_articles(news_summary)
        return Response({
            'input': user_input,
            'newsSummary': news_summary,
            'relatedArticles': related_articles
        })
    else:
        # 기존 Q&A 처리 로직
        save_user_question(user_input, kakao_id)
        #newsdb = Chroma(persist_directory="news_db", embedding_function=embedding_function)
        vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)
        question_prompt = ChatPromptTemplate.from_template("""context를 활용해서 질문에 대답해줘.

<context>
{context}
</context>

질문: {input}""")
        document_chain = create_stuff_documents_chain(llm, question_prompt)
        retriever = vectordb.as_retriever()
        advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        return Response({
            'input': user_input,
            'answer': response.get('answer', 'No answer found'),
            'context': response.get('context', [])
        })
    
def summarize_news_link(news_url):
    try:
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.select_one("div#newsct_article article#dic_area")  # 네이버 뉴스 본문이 담긴 태그 (변경될 수 있음)
        if content:
            article = content.get_text(strip=True)
        else:
            return ''
    
    except requests.exceptions.Timeout:
        print(f"Request to {news_url} timed out.")
        return ''
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ''
    
    response = llm.call_as_llm(f"다음 신문 내용을 요약해줘: {article}")
    return response

def search_related_articles(news_summary, top_k=3):
    vectordb = Chroma(persist_directory="news_db", embedding_function=embedding_function)
    search_results = vectordb.similarity_search(news_summary, k=top_k)
    related_articles = [result.metadata.get("url") for result in search_results]
    return related_articles

def save_user_question(question, kakao_id):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = []

    # 사용자 질문을 Document로 변환
    document = Document(
        page_content=question,
        metadata={"kakaoId": kakao_id}  # 카카오 아이디 메타데이터로 추가
    )

    texts = text_splitter.split_documents([document])
    documents.extend(texts)

    # 기존 user_db 불러오기
    user_db = Chroma(persist_directory="user_db", embedding_function=embedding_function)

    # 새로운 데이터 추가
    user_db.add_documents(documents)

def get_user_query_vector(kakao_id):
    # 사용자 DB에서 사용자 질문 벡터 가져오기
    user_db = Chroma(persist_directory="user_db", embedding_function=embedding_function)
    docs = user_db.get(where={"kakaoId": kakao_id}, include=["embeddings"])

    if not docs or 'embeddings' not in docs:
        return None

    embeddings = np.array(docs['embeddings'])
    user_vector = np.mean(embeddings, axis=0)
    
    return user_vector

def get_relevant_news(request):
    kakao_id = request.GET.get('kakao_id')
    user_vector = get_user_query_vector(kakao_id)
    print("get user vector!!! ", len(user_vector))
    # Chroma DB에서 뉴스 기사 검색
    vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)
    search_results = vectordb.similarity_search_by_vector(embedding=user_vector.tolist(), k=3)
    news_urls = [result.metadata.get("url") for result in search_results]
    news_summary = []
    for url in news_urls:
        news_summary.append(summarize_news_link(url))
    return JsonResponse({'urls': news_urls, 'summary': news_summary})

def get_relevant_users(request):
    query = request.GET.get('query')
    if not query:
        return JsonResponse({"error": "Query parameter is missing"}, status=400)
    
    userdb = Chroma(persist_directory="user_db", embedding_function=embedding_function)
    docs = userdb.similarity_search(query=query, k=3)

    users = []
    for doc in docs:
        users.append(doc.metadata['kakaoId'])

    return JsonResponse({'users': users})

@api_view(['POST'])
def recommend_company(request):
    interest = request.data.get('interest')

    vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)

    prompt = ChatPromptTemplate.from_template("""
    context와 유저 관심분야를 참고해서 가장 관련 깊은 회사를 하나만 추천해줘. 회사명만 짧게 대답해야해.

    <context>
    {context}
    </context>

    유저 관심분야: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectordb.as_retriever()
    advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

    response = retrieval_chain.invoke({"input": interest})
    return Response({
        "input": response['input'],
        "context": response['context'],
        "answer": response['answer']
    })