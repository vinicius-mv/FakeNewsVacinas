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
# random_seed = 19

model_lr.train(X, Y, test_size=0.25, seed=random_seed)
model_dt.train(X, Y, test_size=0.25, seed=random_seed)
model_nb.train(X, Y, test_size=0.25, seed=random_seed)

print("Models trained!")

model_lr.show_info()
model_dt.show_info()
model_nb.show_info()

# Charts
# model_lr.get_confusion_matrix()
# model_dt.get_confusion_matrix()
# model_nb.get_confusion_matrix()
# chart = ChartGenerator.get_dataset_classes_proportion(data)

# Demo predictions using the models
text = "@Oplebeu92 @DanielaAdornoM1 @folha Então já q vc é informado me fale sobre duas situações: já q falou da " \
        "vacina, o que diz do presidente fazer propaganda e comprar um remédio não tem comprovação científica pra " \
        "tratar a covid com dinheiro publico? E qual o plano do governo de combate a pandemia? Sem rodeios"

y0_test_lr = model_lr.predict([text])
print('pred (lr):' + str(y0_test_lr))

y0_test_dt = model_dt.predict([text])
print('pred (dt):' + str(y0_test_dt))

y0_test_nb = model_nb.predict([text])
print('pred (nb):' + str(y0_test_nb))

print(text)
model_lr.analyze_shap_values([text])
# model_nb.analyze_shap_values([text])
# model_dt.analyze_shap_values([text])

print("DONE!")
