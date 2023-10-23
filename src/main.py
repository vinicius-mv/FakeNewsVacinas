import random
import numpy as np
import pandas as pd
import spacy

from utils.MongoConnector import MongoConnector
from core.DecisionTreeModel import DecisionTreeModel
from core.LogisticRegressionModel import LogisticRegressionModel
from core.NaiveBayesModel import NaiveBayesModel
from core.ChartGenerator import ChartGenerator

client = MongoConnector()
client.initialize()

# if missing spacy pt_core_new_sm file - run the following command
# spacy.cli.download("pt_core_news_sm")

# Get data from mongodb
collection = "tweets"
query = {"is_missinginfo": {"$gt": -1}}
cursor = client.find(collection, query)
data = pd.DataFrame(cursor)
data.set_index("_id", inplace=True)

# Initialize prediction models
model_log = LogisticRegressionModel()
model_dtree = DecisionTreeModel()
model_nb = NaiveBayesModel()

X = data['content']
Y = data['is_missinginfo']

print("Training models...")

# random_seed = random.randint(0, 1000)
random_seed = 26 # temp for replicability

model_log.train(X, Y, test_size=0.25, seed=random_seed)
model_dtree.train(X, Y, test_size=0.25, seed=random_seed)
model_nb.train(X, Y, test_size=0.25, seed=random_seed)

print("Models trained!")

model_log.show_info()
model_dtree.show_info()
model_nb.show_info()

ChartGenerator.get_confusion_matrix(model_log)
ChartGenerator.get_confusion_matrix(model_dtree)
ChartGenerator.get_confusion_matrix(model_nb)

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

chart = ChartGenerator.get_dataset_classes_proportion(data)

client.close()

print("DONE!")
