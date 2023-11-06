import random
import numpy as np
import pandas as pd
import shap
import spacy

from utils.MongoConnector import MongoConnector
from core.DecisionTreeModel import DecisionTreeModel
from core.LogisticRegressionModel import LogisticRegressionModel
from core.NaiveBayesModel import NaiveBayesModel
from core.ChartGenerator import ChartGenerator

# if missing spacy pt_core_new_sm file - run the following command
# spacy.cli.download("pt_core_news_sm")

# Get data from mongodb
client = MongoConnector()
client.initialize()
print("Loading data...")
collection = "tweets"
query = {"is_missinginfo": {"$gt": -1}}
cursor = client.find(collection, query)
data = pd.DataFrame(cursor)
print("Data loaded")
client.close()

data.set_index("_id", inplace=True)

# Initialize prediction models
model_lr = LogisticRegressionModel()
model_dt = DecisionTreeModel()
model_nb = NaiveBayesModel()

X = data['content']
Y = data['is_missinginfo']

print("Training models...")

# random_seed = random.randint(0, 1000)
# temp for replicability
random_seed = 3

model_lr.train(X, Y, test_size=0.25, seed=random_seed)
model_dt.train(X, Y, test_size=0.25, seed=random_seed)
model_nb.train(X, Y, test_size=0.25, seed=random_seed)

print("Models trained!")

model_lr.show_info()
model_dt.show_info()
model_nb.show_info()

# Charts
model_lr.get_confusion_matrix()
model_dt.get_confusion_matrix()
model_nb.get_confusion_matrix()
chart = ChartGenerator.get_dataset_classes_proportion(data)

# Demo predictions using the models
text1 =  model_lr.X_test[0]

y0_test_lr = model_lr.predict([text1])
print('x:' + str(y0_test_lr))

y0_test_dt = model_dt.predict([text1])
print('y:' + str(y0_test_dt))

y0_test_nb = model_nb.predict([text1])
print('z:' + str(y0_test_nb))

model_lr.explain_prediction(text1)
# model_dt.explain_prediction(text1)

print("DONE!")
