import pandas as pd
import numpy as np
import os

absolute_path = os.path.dirname(__file__)

df = pd.read_csv('datasets\\vacinas-dataset.csv', index_col='id')

df.head()

updated_rows_df = pd.DataFrame()

for index, row in df.iterrows():
    # tweets already classified
    if (row['is_missinginfo'] >= -1):
        continue

    print(f"Date: '{row.date}'")
    print(f"User: '{row.user_displayname}'")
    print(f"Content: '{row.content}'")
    print()

    print("Important notes:")
    print("* A Anvisa concedeu em 17/01/2021 o registro emergencial da vacina CoronaVac contra a Covid-19")
    print("* A Anvisa concedeu em 23/02/2021 o registro definitivo à vacina contra a covid-19 Pfizer / BioNtech")
    print("* A Anvisa concedeu em 12/03/2021 o registro da vacina da Fiocruz/AstraZeneca\n")
    response = input(
        "\rIs it fake news about covid vacines?\n1 -> yes\n0 -> no\n-1 -> not sure / not related\n * - skip\ne -> exit\nR: ")
    response = response.lower()
    print("\n")
    if (response == "1"):
        df.at[index, 'is_missinginfo'] = 1  # fake
    elif (response == "0"):
        df.at[index, 'is_missinginfo'] = 0  # real
    elif (response == "-1"):
        df.at[index, 'is_missinginfo'] = -1  # not sure / not related
    elif (response == "e"):  # exit
        break

df.head()

df.to_csv('datasets\\vacinas-dataset.csv', mode='w+', index=True)
print("DONE!")
