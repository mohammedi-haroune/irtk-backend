import datetime

from django.db import models
from django.utils import timezone

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text

class Document(object):
    def __init__(self, fileid, sample=None):
        self.fileid = fileid
        self.sample = sample

class VectorModelResult(object):
    def __init__(self, fileid, inner_product=None, dice_coef=None, cosinus_measure=None, jaccard_coef=None):
        self.fileid = fileid
        self.inner_product = inner_product
        self.dice_coef = dice_coef
        self.cosinus_measure = cosinus_measure
        self.jaccard_coef = jaccard_coef
