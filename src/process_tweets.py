from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_absolute_path

absolute_path = get_absolute_path()

data = pd.read_csv(absolute_path + "\\datasets\\vacinas-dataset.csv")

#  filter out rows based on missing values in a column
data = data[data.is_missinginfo.notnull()]

# filter out rows with inconclusive information
data = data[data['is_missinginfo'] >= 0]
data.shape

data.head()

data = data.drop_duplicates(subset=['id'])
data.shape

# Shuffling
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

data.head()
sns.countplot(data=data, x='is_missinginfo',
              order=data['is_missinginfo'].value_counts().index)

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text_data):
    treated_text = []
    for setence in tqdm(text_data):
        # setence = re.sub(r'\?+', '\?', setence)
        # setence = re.sub(r'[^\w\s"?]', '', setence)
        setence = re.sub(r'[^\w\s]', '', setence)
        treated_text.append(' '.join(token.lower()
                                     for token in str(setence).split()
                                     if token not in stopwords.words('portuguese')))
    return treated_text


preprocecessed_review = preprocess_text(data['content'].values)

data['content'] = preprocecessed_review


def plot_word_cloud(words: str) -> None:
    # set wordcloud settings
    wordcloud = WordCloud(width=1600, height=800, random_state=21,
                          max_font_size=110, collocations=False)
    # set plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.generate(words), interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Real wordcloud
real_words = []
for word in data['content'][data['is_missinginfo'] == 0]:
    real_words.append(word)

real_consolidated = ' '.join(real_words)
print("Word Cloud Real Tweets")
plot_word_cloud(real_consolidated)

# Fake wordcloud
fake_words = []
for word in data['content'][data['is_missinginfo'] == 1]:
    fake_words.append(word)

fake_consolidated = ' '.join(fake_words)
print("Word Cloud Fake Tweets")
plot_word_cloud(fake_consolidated)

# Top words bargraph


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = []

    for word, idx in vec.vocabulary_.items():
        words_freq.append([word, sum_words[0, idx]])

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[: n]


print("Top Words")
common_words = get_top_n_words(data['content'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 6),
    xlabel='Top Words',
    ylabel='Count',
    title='Bar Count of Top Words Frequency'
)

# Converting text into Vectors
# Before converting the data into vectors, split it into train and test.

x_train, x_test, y_train, y_test = train_test_split(
    data['content'], data['is_missinginfo'], test_size=0.25)

for content in x_train:
    print(content)
# convert the training data into vectors
vectorization = TfidfVectorizer(strip_accents='ascii', ngram_range=(1, 4))
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

# Model training, Evaluation, and Prediction

# Classifier: Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train, y_train)
log_reg_accuracy_train = accuracy_score(
    y_train, log_reg_model.predict(x_train))
print("LogisticRegression train accuracy: " + str(log_reg_accuracy_train))
log_reg_accuracy_test = accuracy_score(y_test, log_reg_model.predict(x_test))
print("LogisticRegression test accuracy: " + str(log_reg_accuracy_test))

# Classifier: Decision Tree
dec_tree_model = DecisionTreeClassifier()
dec_tree_model.fit(x_train, y_train)
decision_tree_accuracy_train = accuracy_score(
    y_train, dec_tree_model.predict(x_train))
print("DecisionTreeClassifier train accuracy: " +
      str(decision_tree_accuracy_train))
decision_tree_accuracy_test = accuracy_score(
    y_test, dec_tree_model.predict(x_test))
print("DecisionTreeClassifier test accuracy: " +
      str(decision_tree_accuracy_test))


def log_reg_predict(text: str):
    print("\n\nPredicting:\n")
    text = preprocess_text([text])
    print(text[0])
    x_new = vectorization.transform(text)
    y_new = log_reg_model.predict_proba(x_new)
    # print("X=%s, Predicted reg log=%s" % (x_new[0], y_new_log[0]))
    # y_new_log[0][0] --> chance to be 0 (is_missing_info = false)
    # y_new_log[0][1]  --> chance to be 1 (is_missing_info = true)
    print("Log Reg predicted true prob=%s" % (y_new[0][0]))
    print("Log Reg predicted fake prob=%s" % (y_new[0][1]))
    y_new_final = log_reg_model.predict(x_new)
    print(f"Predicted reg log = {y_new_final[0]}")
    return


def dec_tree_predict(text: str):
    print("\n\nPredicting:\n")
    text = preprocess_text([text])
    print(text[0])
    x_new = vectorization.transform(text)
    y_new = dec_tree_model.predict_proba(x_new)
    # y_new[0][0]  --> chance to be 0 (is_missing_info = false)
    # y_new_log[0][1]  --> chance to be 1 (is_missing_info = true)
    print("Dec Tree predicted true prob=%s" % (y_new[0][0]))
    print("Dec Tree predicted fake prob=%s" % (y_new[0][1]))
    y_new_final = dec_tree_model.predict(x_new)
    print(f"Predicted reg log = {y_new_final[0]}")
    return
