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
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever
from datetime import datetime, timedelta, date
import os

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