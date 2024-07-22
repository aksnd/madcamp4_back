from django.db import models

class Item(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name

class User(models.Model):
    kakao_id = models.CharField(max_length=100)
    nickname = models.CharField(max_length=100)
    
    def __str__(self):
        return self.kakao_id