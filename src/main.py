import random
import numpy as np
import pandas as pd

from utils.MongoConnector import MongoConnector
from core.DecisionTreeModel import DecisionTreeModel
from core.LogisticRegressionModel import LogisticRegressionModel
from core.NaiveBayesModel import NaiveBayesModel


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

print("Training models...")

random_seed = random.randint(0, 1000)

model_log.train(X, Y, test_size=0.25, seed=random_seed)
model_dtree.train(X, Y, test_size=0.25, seed=random_seed)
model_nb.train(X, Y, test_size=0.25, seed=random_seed)

print("Models trained!")

model_log.show_info()
model_dtree.show_info()
model_nb.show_info()

# Demo predictions using the models
text1 = "@Oplebeu92 @DanielaAdornoM1 @folha Então já q vc é informado me fale sobre duas situações: já q falou da " \
        "vacina, o que diz do presidente fazer propaganda e comprar um remédio não tem comprovação científica pra " \
        "tratar a covid com dinheiro publico? E qual o plano do governo de combate a pandemia? Sem rodeios"
        
x = model_log.predict([text1])
print('x:' + str(x))
y = model_dtree.predict([text1])
print('y:' + str(y))
z = model_nb.predict([text1])
print('z:' + str(z))

client.close()

print("DONE!")