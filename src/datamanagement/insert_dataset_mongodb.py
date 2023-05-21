import sys
sys.path.insert(0,"..")

import pandas as pd

from utils.MongoConnector import MongoConnector
from utils.path import get_main_path

path = get_main_path()

df = pd.read_csv(path + "\\datasets\\vacinas-dataset.csv")
df.drop_duplicates(subset=["id"], inplace=True)
df.rename(columns={"id": "_id"}, inplace=True)

data = df.to_dict("records")

collection = "tweets"

client = MongoConnector()
client.initialize()

client.insert_many(collection, data)
