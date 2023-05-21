import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from core.PredictionModel import PredictionModel


# https://www.kaggle.com/code/lykin22/twitter-sentiment-analysis-with-naive-bayes-85-acc
class NaiveBayesModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model = ComplementNB()

    @staticmethod
    def preprocess_text(text):
        return PredictionModel.preprocess_text(text)

    def train(self, x, y, test_size=0.25):
        super().train(x, y, test_size)

    def show_info(self):
        print("Naive Bayes Model:")
        super().show_info()

    def model_score(self, x, y):
        return super().model_score(x, y)

    def predict(self, tweets):
        return super().predict(tweets)


