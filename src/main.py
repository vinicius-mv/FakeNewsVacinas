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
print(f"Data loaded - shape {data.shape}")
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
random_seed = 17

model_lr.train(X, Y, test_size=0.25, seed=random_seed)
model_dt.train(X, Y, test_size=0.25, seed=random_seed)
model_nb.train(X, Y, test_size=0.25, seed=random_seed)

print("Models trained!")

model_lr.dataset_info()

model_lr.model_info()
model_dt.model_info()
model_nb.model_info()
print("\n\n")

# Charts
model_lr.get_confusion_matrix()
model_dt.get_confusion_matrix()
model_nb.get_confusion_matrix()
chart = ChartGenerator.get_dataset_classes_proportion(data)

# Demo predictions using the models

# message1 - label: 1
message_fake0 = "@luislacombereal Eu também escolhi não tomar,estou tomando os cuidados mas sem neuras,porém escolhi não ser cobaia de vacina nenhuma!"
message_fake1 = "A Satânica vermelha queria aplicar vacina sem comprovação da Anvisa? Esgoto, o tietê é mais limpo do que essa porca.Depois vem com narrativa que somos odiosos. Olha o que vocês  façam filhos satânico, vivem 24 hora insultando a inteligência com essa mentiras"
message_fake2 = "A verdade , como eu sempre pensei, estão fazendo o povo de cobaia no caso da vacina  Coronovac, pois as duas doses não estão sendo suficientes para imunização, já estão decidindo dar a 3ª dose da vacina. Isso é que eu chamo vacina Xing Ling 🤣🤣🤣"
message_fake3 = "@OmarAzizSenador vacina de 49%. Eu tinha vergonha de sentar numa mesa pra isso. Fazendo o brasileiro de cobaia."
message_fake4 = "@SLC1959 @OmarAzizSenador @opovoonline Na China não usam a Coronavac e na Europa é proibido entrar quem fez está vacina devido a baixa Eficácia. E a Eficácia só foi definida em janeiro. Queriam que os brasileiros fossem cobaia? Daí não."
message_fake5 = "@UOLPolitica Dimas Covas disse: Uso emergencial das vacinas mas esqueceu de mencionar que a aprovação do uso emergencial não ISENTA a vacina de ser APROVADA pela ANVISA! OU SEJA, sem a comprovação da vacina pela ANVISA não poderia começar a vacinação https://t.co/8zQDqePMkr"
message_fake6 = "Ain, vacina experimental, tudo bem, pq PODE fazer a vida voltar ao normal. Ain, remédio sem comprovação científica, de jeito nenhum, pq PODE causar disritmia. Hipocritas! NUNCA foi pra \"salvar vidas\", SEMPRE foi pelos intere$$e$ inconfessáveis das BigFarmas."
message_fake7 = "@mriovi @Leolabar @PaganiBibi @MauricioRicardo @jairbolsonaro @minsaude A vacina mais rápida a ficar pronta foi da caxumba, 4 anos, essas 4 meses, vai cobaia!"

message_real0 = "@zen4rt @OBQDC @carolinabf Então não entendi o ponto de comprar um remedio sem comprovação pra doença e não uma vacina."
message_real1 = "@sylberman @radioitatiaia para de indicar remédio sem comprovação científica a única que trata o corona vírus e a VACINA!"
message_real2 = "Na cidade onde nasci e onde meus pais moram 100% fos leitos ocupados grande ameaça de desabastecimento de oxigênio nos próximos dias tudo por que um governo genocida não quis comprar vacina.#ForaBolsonaro #VacinaJa #Genocida #VacinaSalvaVidas"
message_real3 = "@Oplebeu92 @DanielaAdornoM1 @folha Então já q vc é informado me fale sobre duas situações: já q falou da vacina, o que diz do presidente fazer propaganda e comprar um remédio não tem comprovação científica pra tratar a covid com dinheiro publico? E qual o plano do governo de combate a pandemia? Sem rodeios"
message_real4 = "Cês já pararam pra pensar que o governo deve ter experimentado a imunidade de rebanho na população? Por isso a recusa da vacina. Por isso o tratamento precoce (que favorece um pessoalzinho aí) e também as campanhas anti-prevenção. A gente foi cobaia. Cobaia igual rato."
message_real5 = "Não tenho palavras. Diante deste negacionismo institucional, assim compreendi; o jovem\trabalhador\honesto\pagador de imposto está no rabo da fila da vacina, para servir de cobaia a imunização de rebanho. E ninguém aponta isto? Quantos jovens morreram? #VacinaAosJovens"
message_real6 = "Cês já pararam pra pensar que o governo deve ter experimentado a imunidade de rebanho na população? Por isso a recusa da vacina. Por isso o tratamento precoce (que favorece um pessoalzinho aí) e também as campanhas \"anti-prevenção\". A gente foi cobaia. Cobaia igual rato."

y0_test_lr = model_lr.predict([message_fake1])
print('pred (lr):' + str(y0_test_lr))

y0_test_dt = model_dt.predict([message_fake1])
print('pred (dt):' + str(y0_test_dt))

y0_test_nb = model_nb.predict([message_fake1])
print('pred (nb):' + str(y0_test_nb))

model_lr.explainer_summary_plot(model_lr.X_train)
model_lr.explainer_beeswarm_plot(model_lr.X_train)

# model_lr.explainer_detailed_plot(message_fake1)

print("DONE!")
