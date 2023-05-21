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
        # # Preprocess tweets
        # x_preprocessed = [self.preprocess_text(tweet) for tweet in x]

        # # Split data between train and test
        # x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=test_size)

        # # Set train size control fields
        # self.total_tweets_train = len(x)
        # self.fake_tweets_train = PredictionModel.get_fake_tweets_length(y_train)

        # # Create TF-IDF vectorizer
        # # self.vectorizer = TfidfVectorizer()
        # self.vectorizer = TfidfVectorizer(strip_accents="ascii", ngram_range=(1, 2))
        # x_tfidf = self.vectorizer.fit_transform(x_train)
        # self.model.fit(x_tfidf, y_train)

        # # Calculate model accuracy
        # self.model_train_accuracy = self.model_score(x_train, y_train)
        # self.model_test_accuracy = self.model_score(x_test, y_test)

    def show_info(self):
        print("Gaussian Naive Bayes Model:")
        super().show_info()

    def model_score(self, x, y):
        return super().model_score(x, y)

    def predict(self, tweets):
        return super().predict(tweets)


