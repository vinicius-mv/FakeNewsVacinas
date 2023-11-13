import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.core.PredictionModel import PredictionModel


# https://www.geeksforgeeks.org/fake-news-detection-using-machine-learning/
class DecisionTreeModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Decision Tree"
        self.model = DecisionTreeClassifier()

    def explainer_summary_plot(self, tweet):
        raise NotImplemented()

    def explainer_beeswarm_plot(self, tweet):
        raise NotImplemented()

