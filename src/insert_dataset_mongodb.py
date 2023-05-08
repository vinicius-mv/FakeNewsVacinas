import pandas as pd

from MongoConnector import MongoConnector
from utils import get_absolute_path

absolute_path = get_absolute_path()

df = pd.read_csv(absolute_path + "\\datasets\\vacinas-dataset.csv")
df.drop_duplicates(subset=["id"], inplace=True)
df.rename(columns={"id": "_id"}, inplace=True)

data = df.to_dict("records")

collection = "tweets"

client = MongoConnector()
client.initialize()

client.insert_many(collection, data)
