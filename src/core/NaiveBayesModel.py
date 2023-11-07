import numpy as np
import pandas as pd
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from src.core.PredictionModel import PredictionModel


# https://www.kaggle.com/code/lykin22/twitter-sentiment-analysis-with-naive-bayes-85-acc
class NaiveBayesModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Naive Bayes"
        self.model = GaussianNB()

    @staticmethod
    def preprocess_text(text):
        return PredictionModel.preprocess_text(text)

    def train(self, X, Y, test_size, seed=19):
        super().train(X, Y, test_size, seed, need_dense_data=True)

    def show_info(self):
        super().show_info()

    def predict(self, tweets):
        return super().predict(tweets)

    def analyze_shap_values(self, tweets):
        # Once SHAP does not have a specific implementation for NaiveBayes
        # The Kernel is an explainer for probabilist models and provide and estimated value for NaiveBayes
        self.explainer = shap.Explainer(self.model.predict, self.X_train_tfidf)
        shap_values = self.explainer.shap_values(self.X_train_tfidf)
        shap.summary_plot(shap_values, self.X_train_tfidf,
                          feature_names=self.vectorizer.get_feature_names_out())



