import requests
from newspaper import Article
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from .get_emotion import summarize_score
from openai import OpenAI

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"

def get_news_articles(query, max_results=1000):
    headers = {
        'X-Naver-Client-Id': os.environ.get('NAVER_CLIENT_ID'),
        'X-Naver-Client-Secret': os.environ.get('NAVER_CLIENT_SECRET')
    }
    articles = []
    params = {
        'query': query,
        'display': 100,  # 한 페이지에 최대 100개 기사
        'start': 1,
        'sort': 'sim'
    }
    
    while len(articles) < max_results:
        response = requests.get(NAVER_API_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            if not items:
                break
            articles.extend(items)
            params['start'] += 100  # 다음 페이지
        else:
            break

    return articles[:max_results]  # 요청한 최대 수만큼 자르기



def fetch_article_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.select_one("div#newsct_article article#dic_area")  # 네이버 뉴스 본문이 담긴 태그 (변경될 수 있음)
        return content.get_text(strip=True) if content else ''
    
    except requests.exceptions.Timeout:
        print(f"Request to {url} timed out.")
        return ''
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ''


def save_news_articles(articles, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    api_key = os.environ.get('OPENAI_API_KEY')
    embedding_function = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    documents = []

    for article in articles:
        url = article['link']
        content = fetch_article_content(url)
        print("뉴스 본문:   ", content)
        title = article['title']
        date = datetime.strptime(article['pubDate'], '%a, %d %b %Y %H:%M:%S %z').date()

        document_text = f"{title}\n{content}"
        document = Document(
            page_content = document_text,
            metadata = {"date": date.strftime('%Y-%m-%d'), "url": url, "company": query}
        )

        texts = text_splitter.split_documents([document])
        documents.extend(texts)

    # 기존 Chroma 데이터베이스 불러오기
    vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)

    # 새로운 데이터 추가
    vectordb.add_documents(documents)

def crawl_and_save_news(query):
    articles = get_news_articles(query)
    save_news_articles(articles, query)
    
    
    
def get_company_news_today(company, date):
    api_key = os.environ.get('OPENAI_API_KEY')
    embedding_function = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    vectordb = Chroma(persist_directory="company_db", embedding_function=embedding_function)
    docs = vectordb.get(where={"company": company})
    docs_metadatas = docs['metadatas']
    docs_documents = docs['documents']
    article_link_list =[]
    for metadata in docs_metadatas:
        if(metadata['date']=='2024-07-18' and (metadata['url'] not in article_link_list)):
            article_link_list.append(metadata['url'])
    if(len(article_link_list)>20):
        article_link_list = article_link_list[0:20]
    filtered_articles = []
    for link in article_link_list:
        docs = vectordb.get(where={"url": link})
        article = {'content':'', 'link': link}
        for documents in docs['documents']:
            article['content'] = article['content']+documents
        article['summary'] = summarize_score(article['content'])
        filtered_articles.append(article)    
    return filtered_articles
    # 네이버 API를 이용해 뉴스 기사 가져오기
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={company}&display=20&sort=sim"  # 최대 20개 기사 가져오기
    headers = {
        'X-Naver-Client-Id': os.environ.get('NAVER_CLIENT_ID'),
        'X-Naver-Client-Secret': os.environ.get('NAVER_CLIENT_SECRET')
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        articles = response.json()['items']
        filtered_articles = []
        for article in articles:
            pub_date = datetime.strptime(article['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
            if pub_date.date() == date:
                article['content']= fetch_article_content(article['link'])
                filtered_articles.append(article)
        return filtered_articles
    else:
        return []
    
def get_input_news_today(company,date):
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={company}&display=20&sort=sim"  # 최대 20개 기사 가져오기
    headers = {
        'X-Naver-Client-Id': os.environ.get('NAVER_CLIENT_ID'),
        'X-Naver-Client-Secret': os.environ.get('NAVER_CLIENT_SECRET')
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        articles = response.json()['items']
        filtered_articles = []
        for article in articles:
            pub_date = datetime.strptime(article['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
            if pub_date.date() == date:
                article['content']= fetch_article_content(article['link'])
                filtered_articles.append(article)
        return filtered_articles
    else:
        return []

