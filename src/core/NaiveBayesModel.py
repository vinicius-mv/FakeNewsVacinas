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

    def explainer_summary_plot(self, tweet):
        raise NotImplemented()

    def explainer_beeswarm_plot(self, tweet):
        raise NotImplemented()


