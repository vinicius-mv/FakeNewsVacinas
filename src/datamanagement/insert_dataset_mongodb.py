import sys
sys.path.insert(0,"..")

import pandas as pd

from utils.MongoConnector import MongoConnector

df = pd.read_csv("..\\datasets\\vacinas-dataset.csv")
df.shape
df.drop_duplicates(subset=["_id"], inplace=True)
df.shape
df.rename(columns={"id": "_id"}, inplace=True)

data = df.to_dict("records")

collection = "tweets"

client = MongoConnector()
client.initialize()

client.insert_many(collection, data)
