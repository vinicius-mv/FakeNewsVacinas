import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.core.PredictionModel import PredictionModel


# https://www.geeksforgeeks.org/fake-news-detection-using-machine-learning/
class DecisionTreeModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Decision Tree"
        self.model = DecisionTreeClassifier()

    def explain_prediction(self, tweet):
        raise NotImplemented()

