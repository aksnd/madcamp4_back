import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"
NAVER_CLIENT_ID = 'ToJzWzJttiqDu376Fj3z'
NAVER_CLIENT_SECRET = 'MXCA9OQhyH'

def get_news_articles(query, start_date, end_date):
    headers = {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET,
    }
    articles = []
    for date in [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]:
        params = {
            'query': query,
            'display': 10,
            'start': 1,
            'sort': 'sim',
            'start_date': date.strftime('%Y-%m-%d'),
            'end_date': date.strftime('%Y-%m-%d')
        }
        response = requests.get(NAVER_API_URL, headers=headers, params=params)
        if response.status_code == 200:
            articles.extend(response.json().get('items', []))
    return articles

def fetch_article_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.select_one("div#newsct_article article#dic_area")  # 네이버 뉴스 본문이 담긴 태그 (변경될 수 있음)
        return content.get_text(strip=True) if content else ''
    
    return ''

def save_news_articles(articles):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
    embedding_function = SentenceTransformerEmbeddings(model)
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
            metadata = {"date": date.strftime('%Y-%m-%d')}
        )

        texts = text_splitter.split_documents([document])
        documents.extend(texts)

    Chroma.from_documents(documents=documents, embedding=embedding_function, persist_directory="chroma_db")


def crawl_and_save_news(query, start_date, end_date):
    articles = get_news_articles(query, start_date, end_date)
    save_news_articles(articles)