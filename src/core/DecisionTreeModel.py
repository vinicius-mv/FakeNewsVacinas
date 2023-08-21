import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from core.PredictionModel import PredictionModel


# https://www.geeksforgeeks.org/fake-news-detection-using-machine-learning/
class DecisionTreeModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Decision Tree"
        self.model = DecisionTreeClassifier()
        
    @staticmethod
    def preprocess_text(text):
        return PredictionModel.preprocess_text(text)

    def train(self, x, y, test_size, seed=19):
        super().train(x, y, test_size, seed)

    def show_info(self):
        super().show_info()

    def model_score(self, x, y):
        return super().model_score(x, y)

    def predict(self, tweets):
        return super().predict(tweets)
