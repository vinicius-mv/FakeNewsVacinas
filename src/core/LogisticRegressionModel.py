import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.PredictionModel import PredictionModel


# https://www.geeksforgeeks.org/fake-news-detection-using-machine-learning/
# https://realpython.com/logistic-regression-python/
class LogisticRegressionModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Logistic Regression"
        self.model = LogisticRegression()

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
        super()._explain_prediction(tweet)
