import sys

from src.utils.MongoConnector import MongoConnector

sys.path.insert(0,"..")

import datetime as dt
import os
import pandas as pd


path = os.getcwd()

client = MongoConnector()
client.initialize()

# Get data from mongodb
collection = "tweets"
query = {}
cursor = client.find(collection, query)
data = pd.DataFrame(cursor)
data.set_index("_id", inplace=True)

now = dt.datetime.now()
date = now.strftime("%m-%d-%Y")

data.to_csv(f"{path}\\bkps\\tcc_tweets_{date}_bkp.csv")

client.close()

print("Backup completed")