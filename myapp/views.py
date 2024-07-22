from rest_framework import viewsets
from .models import Item
from .serializers import ItemSerializer
from django.http import JsonResponse
from django.views import View
from .yfinance_control import get_stock_price_on_date, get_stock_price
from .crawling import get_company_news_today
from .get_emotion import calculate_emotion_score, calculate_relevance_score
import yfinance as yf
from rest_framework.decorators import api_view
from rest_framework.response import Response
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

import requests
from django.shortcuts import redirect, render
from django.contrib.auth import login
#from django.contrib.auth.models import User
from .models import User  # 커스텀 User 모델 임포트

def kakao_login(request):
    app_rest_api_key = '619b59af6b0f7c4accad132301921e47'
    redirect_uri = 'http://localhost:8000/kakao/callback/'
    kakao_oauth_url = f"https://kauth.kakao.com/oauth/authorize?client_id={app_rest_api_key}&redirect_uri={redirect_uri}&response_type=code"
    return redirect(kakao_oauth_url)

def kakao_callback(request):
    app_rest_api_key = '619b59af6b0f7c4accad132301921e47'
    redirect_uri = 'http://localhost:8000/kakao/callback/'
    code = request.GET.get('code')
    print(code)
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
    return redirect('http://localhost:3000/')

class PredictView(View):
    def get(self, request):
        company = request.GET.get('company', None)
        if not company:
            return JsonResponse({'error': 'company is required'}, status=400)
        dates = date.today()
        try:
            price = get_stock_price(company,dates)
            articles = get_company_news_today(company,dates)

            result_articles = []
            for article in articles:
                if(article['content']!=''):
                    result_articles.append({
                    'title': article['title'],
                    'emotion': calculate_emotion_score(article['content']),
                    'relevance': calculate_relevance_score(article['content'],company),
                    'link': article['link']
                    })
                else:
                    print("cannot get contents")
                    result_articles.append({
                        'title': article['title'],
                        'emotion': calculate_emotion_score(article['title']),
                        'relevance': calculate_relevance_score(article['title'],company),
                        'link': article['link']
                    })

            response_data =  {
                'price': price,
                'articles': result_articles
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
            articles = get_company_news_today(company,dates)
            result_articles = []
            for article in articles:
                if(article['content']!=''):
                    result_articles.append({
                    'title': article['title'],
                    'emotion': calculate_emotion_score(article['content']),
                    'relevance': calculate_relevance_score(article['content'],company),
                    'link': article['link']
                    })
                else:
                    print("cannot get contents")
                    result_articles.append({
                        'title': article['title'],
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

    # 사용자 질문 임베딩 및 저장
    save_user_question(user_input)

    # VectorDB 불러오기
    vectordb = Chroma(persist_directory="news_db", embedding_function=embedding_function)

    # 프롬프트 설정
    question_prompt = ChatPromptTemplate.from_template("""context를 활용해서 질문에 대답해줘.

<context>
{context}
</context>

질문: {input}""")

    # Document Chain 생성
    document_chain = create_stuff_documents_chain(llm, question_prompt)

    # Retriever 설정
    retriever = vectordb.as_retriever()
    advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    # Retrieval Chain 생성
    retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

    # 질의에 대한 응답 생성
    response = retrieval_chain.invoke({"input": user_input})

    # 응답 반환
    return Response({
        'input': user_input,
        'answer': response.get('answer', 'No answer found'),
        'context': response.get('context', [])
    })

def save_user_question(question):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = []

    # 사용자 질문을 Document로 변환
    document = Document(
        page_content=question,
        metadata={"user_id": "admin"}
    )

    texts = text_splitter.split_documents([document])
    documents.extend(texts)

    # 기존 user_db 불러오기
    user_db = Chroma(persist_directory="user_db", embedding_function=embedding_function)

    # 새로운 데이터 추가
    user_db.add_documents(documents)

@api_view(['POST'])
def recommend_company(request):
    interest = request.data.get('interest')

    vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)

    prompt = ChatPromptTemplate.from_template("""
    Based on the following articles, recommend a company that matches the user's interest:

    <context>
    {context}
    </context>

    User's Interest: {input}
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