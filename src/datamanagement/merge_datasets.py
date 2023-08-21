import sys
sys.path.insert(0,"..")

import pandas as pd
import os

from utils.MongoConnector import MongoConnector

files_path = f"..\\datasets\\"

# files already merged: 
# 1. 'vacina sem eficácia-tweets.csv', 
# 2. 'vacina sem comprovação-tweets.csv', 
# 3. 'vacina cobaia-tweets.csv
# 4. 'vacinas-tweets.csv
files_to_read = ['vacina-tweets.csv']

final_path = files_path + "vacinas-dataset.csv"

client = MongoConnector()
client.initialize()

collection = "tweets"
query = {}
cursor = client.find(collection, query)

df = pd.DataFrame(cursor)
df.head()
df.set_index("_id", inplace=True)

for f in os.listdir(files_path):
    if f not in files_to_read:
        continue
    print(f)
    current_file_path = files_path + f
    temp_df = pd.read_csv(current_file_path, index_col="id")
    temp_df.rename(columns={'id': '_id'}, inplace=True)
    df = pd.concat([df, temp_df], axis=0)

# filter portuguese messages only
df = df[df["lang"] == "pt"]
df.shape

print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)

df.head()
df.to_csv("vacinas-dataset.csv", mode="w+", index=True)

print(f"temp datasets merged into file {final_path}")
print("DONE!")
