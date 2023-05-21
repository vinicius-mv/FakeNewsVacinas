import numpy as np
import pandas as pd

from core.PredictionModel import PredictionModel
from core.DecisionTreeModel import DecisionTreeModel
from core.LogisticRegressionModel import LogisticRegressionModel
from core.NaiveBayesModel import NaiveBayesModel
from utils.MongoConnector import MongoConnector


client = MongoConnector()
client.initialize()

# Get data from mongodb
collection = "tweets"
query = {"is_missinginfo": {"$gt": -1}}
cursor = client.find(collection, query)
data = pd.DataFrame(cursor)
data.set_index("_id", inplace=True)

# Initilize prediction models
model_log = LogisticRegressionModel()
model_dtree = DecisionTreeModel()
model_nb = NaiveBayesModel()

X = data['content']
Y = data['is_missinginfo']

model_log.train(X, Y, test_size=0.33)
model_dtree.train(X, Y, test_size=0.33)
model_nb.train(X, Y, test_size=0.33)

model_log.show_info()
model_dtree.show_info()
model_nb.show_info()

# Demo predictions using the models
text1 = "@Oplebeu92 @DanielaAdornoM1 @folha Então já q vc é informado me fale sobre duas situações: já q falou da " \
        "vacina, o que diz do presidente fazer propaganda e comprar um remédio não tem comprovação científica pra " \
        "tratar a covid com dinheiro publico? E qual o plano do governo de combate a pandemia? Sem rodeios"
        
x = model_log.predict([text1])
y = model_dtree.predict([text1])
z = model_nb.predict([text1])

client.close()

print("DONE!")