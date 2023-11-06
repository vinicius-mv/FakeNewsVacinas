import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split

from src.core.PredictionModel import PredictionModel


# https://www.kaggle.com/code/lykin22/twitter-sentiment-analysis-with-naive-bayes-85-acc
class NaiveBayesModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Naive Bayes"
        self.model = ComplementNB()

    @staticmethod
    def preprocess_text(text):
        return PredictionModel.preprocess_text(text)

    def train(self, X, Y, test_size, seed=19):
        super().train(X, Y, test_size, seed)

    def show_info(self):
        super().show_info()

    def predict(self, tweets):
        return super().predict(tweets)

    def explain_prediction(self, tweet):
        raise NotImplemented()


