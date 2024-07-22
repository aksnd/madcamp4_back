from django.core.management.base import BaseCommand
from myapp.crawling import crawl_and_save_news
from datetime import datetime

class Command(BaseCommand):
    help = 'Crawl news and save to database'

    def handle(self, *args, **kwargs):
        crawl_and_save_news('LG에너지솔루션')  # 크롤링 함수 호출
        self.stdout.write(self.style.SUCCESS('Successfully crawled news'))