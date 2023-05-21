import pandas as pd

from src.core.LogisticRegressionModel import LogisticRegressionModel
from utils.MongoConnector import MongoConnector
from core.PredictionModel import PredictionModel

client = MongoConnector()
client.initialize()

# Get data from mongodb
collection = "tweets"
query = {"is_missinginfo": {"$gt": -1}}
cursor = client.find(collection, query)

data = pd.DataFrame(cursor)
data.set_index("_id", inplace=True)

model = LogisticRegressionModel()

model.train(data['content'], data['is_missinginfo'], test_size=0.20)

model.show_info()

text1 = "@Oplebeu92 @DanielaAdornoM1 @folha Então já q vc é informado me fale sobre duas situações: já q falou da " \
        "vacina, o que diz do presidente fazer propaganda e comprar um remédio não tem comprovação científica pra " \
        "tratar a covid com dinheiro publico? E qual o plano do governo de combate a pandemia? Sem rodeios"
text2 = "@jornaldarecord Nem a própria China faz uso da vachina que vendem uma fraudes a coronga do subversivo João " \
        "Doria cedo ou tarde vai acabar na cadeia por estelionato em massa assassinatos em massa"

y = model.predict([text1])

print("DONE")
