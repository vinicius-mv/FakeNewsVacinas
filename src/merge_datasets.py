import pandas as pd
import os

from utils.path import get_absolute_path

absolute_path = get_absolute_path()

# files already merged vacina sem eficácia-tweets.csv', 'vacina sem comprovação-tweets.csv', 'vacina cobaia-tweets.csv
files_to_read = []

final_path = absolute_path + "\\vacinas-dataset.csv"

df = pd.read_csv(final_path, index_col="id")

for f in os.listdir(absolute_path + "\\temp"):
    print(f)
    if f not in files_to_read:
        continue
    temp_path = absolute_path + "\\temp\\" + f
    temp_df = pd.read_csv(temp_path, index_col="id")
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
