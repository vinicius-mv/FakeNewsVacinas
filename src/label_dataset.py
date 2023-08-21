import sys
sys.path.insert(0,"..")

import pandas as pd
import numpy as np

from utils.MongoConnector import MongoConnector

client = MongoConnector()
client.initialize()

collection = "tweets"
query = {"is_missinginfo": np.NaN}
cursor = client.find(collection, query)

df = pd.DataFrame(cursor)
df.head()
df.set_index("_id", inplace=True)

# randomize sample
data = df.sample(frac=1)

updated_rows_df = pd.DataFrame()


def update_row(row):
    query = {"_id": row["_id"]}
    post = {"$set": {"is_missinginfo": row["is_missinginfo"]}}
    client.update_one(collection, query, post)


for index, row in data.iterrows():
    # tweets already classified
    if row["is_missinginfo"] >= -1:
        continue

    row["_id"] = index
    print(f"_id: {row._id}")
    print(f"Date: '{row.date}'")
    print(f"User: '{row.user_displayname}'")
    print(f"Content: '{row.content}'")
    print()

    print("Important notes:")
    print(
        "* A Anvisa concedeu em 2021-01-17 o registro emergencial da vacina CoronaVac"
    )
    print(
        "* A Anvisa concedeu em 2021-01-17 o registro emergencial da vacina Fiocruz/AstraZeneca"
    )
    print(
        "* A OMS concedeu em 2020-12-31 a autorização emergêncial da vacina Pfizer BioNTech"
    )
    print(
        "* A Anvisa concedeu em 2021-02-23 o registro definitivo da vacina Pfizer / BioNtech"
    )
    response = input(
        "Is it fake news about covid vacines?\n1 -> yes\n0 -> no\n-1 -> not sure / not related\n * - skip\ne -> exit\nR: "
    )
    response = response.lower()
    print("\n")
    if response == "1":
        row["is_missinginfo"] = 1  # fake
        update_row(row)
    elif response == "0":
        row["is_missinginfo"] = 0  # real
        update_row(row)
    elif response == "-1":
        df.at[index, "is_missinginfo"] = -1  # not sure / not related
        update_row(row)
    elif response == "e":  # exit
        break
    
client.close()

print("DONE!")
