from django.core.management.base import BaseCommand
from myapp.crawling import crawl_and_save_news
from datetime import datetime

class Command(BaseCommand):
    help = 'Crawl news and save to database'

    def handle(self, *args, **kwargs):
        crawl_and_save_news('삼성전자', datetime(2023, 12, 25), datetime(2023, 12, 26))  # 크롤링 함수 호출
        self.stdout.write(self.style.SUCCESS('Successfully crawled news'))