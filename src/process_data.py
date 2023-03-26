import pandas as pd
import os

absolute_path = os.path.dirname(__file__)

files_to_read = ['vachina-tweets.csv', 'vacina china-tweets.csv', 'vacina cobaia-tweets.csv',
                 'vacina sem comprovação-tweets.csv', 'vacina sem eficácia-tweets.csv']

df = pd.DataFrame()

for f in os.listdir(absolute_path + '\\temp'):
    print(f)
    if (f not in files_to_read):
        continue

    temp_df = pd.read_csv(absolute_path + '\\temp\\' + f)
    df = pd.concat([df, temp_df], axis=0)


df[df['is_missinginfo'] == 1]
df.shape
df.head()

# filter portuguese only messages
df = df[df['lang'] == 'pt']
df.shape

# filter is_missing info not
#df_final = df[df['is_missinginfo'].notnull()]
# df_final.shape
# df.head()

df = df.sample(frac=1)

print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)

df.to_csv('raw-vacinas-dataset.csv', index=False)

print("DONE!")
