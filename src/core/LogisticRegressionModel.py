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
