import re
import nltk
import spacy
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# spacy.cli.download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")

nltk.download('punkt')
nltk.download('stopwords')


class PredictionModel(object):

    def __init__(self):
        self.model = None
        self.model_name = None
        self.vectorizer = None
        self.total_fake_tweets = None
        self.total_tweets = None
        self.accuracy_score = None
        self.precision_score = None
        self.recall_score = None
        self.f1_score = None
        self.crosstab = None

    @staticmethod
    def preprocess_text(text):
        # Remove multiples questions mark
        text = re.sub(r"\?+", "?", text)

        # Filter symbols, punctuation, etc.
        text = re.sub(r"[^\w\s?\"]", "", text)

        # Convert abreviations
        text = re.sub(r"\spq\s", " porque ", text)
        text = re.sub(r"\sq\s", " que ", text)
        text = re.sub(r"\svc\s", " você ", text)
        text = re.sub(r"\smsm\s", " mesmo ", text)

        # Lemmatize tokens and remove stopwords 
        stop_words = set(stopwords.words('portuguese'))
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in stop_words]

        # create a string from tokens
        preprocessed_text = ' '.join(tokens)

        # remove multiple whitespaces
        preprocessed_text = re.sub("\s+", " ", preprocessed_text)

        return preprocessed_text

    @staticmethod
    def get_fake_tweets_length(Y: list):
        total = 0
        y: int
        for y in Y:
            if (int(y) == 1):
                total = total + 1
        return total

    def train(self, x, y, test_size, seed=19):
        # Preprocess tweets
        x_preprocessed = [self.preprocess_text(tweet) for tweet in x]

        # Split data between train and test
        x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=test_size, random_state=seed)

        # Set train size control fields
        self.total_tweets = len(x)
        self.total_fake_tweets = self.get_fake_tweets_length(y)

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(strip_accents="ascii", ngram_range=(1, 1))
        x_tfidf = self.vectorizer.fit_transform(x_train)

        # Train model
        self.model.fit(x_tfidf, y_train)

        # model scores
        self.calculate_scores(x_test, y_test)

    def show_info(self):
        print(f"Model {self.model_name}")
        print('Model - total tweets (train): ' + str(self.total_tweets))
        print('Model - fake tweets (train): ' + str(self.total_fake_tweets))
        print('Model - accuracy score: ' + str(self.accuracy_score))
        print('Model - precision score: ' + str(self.precision_score))
        print('Model - recall score: ' + str(self.recall_score))
        print('Model - f1 score: ' + str(self.f1_score))
        print()

    def predict(self, tweets):
        # Preprocess tweets
        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]

        # Vectorize tweets
        x_tfidf = self.vectorizer.transform(preprocessed_tweets)

        # Make predictions
        predictions = self.model.predict(x_tfidf)

        return predictions

    def calculate_scores(self, x_test, y_test):
        x_tfidf = self.vectorizer.transform(x_test)
        y_pred = self.model.predict(x_tfidf)

        self.crosstab = pd.crosstab(y_pred, y_test)
        TP = self.crosstab[1][1]  # true positives
        TN = self.crosstab[0][0]  # true negatives
        FP = self.crosstab[0][1]  # false positives
        FN = self.crosstab[1][0]  # false negatives
        self.accuracy_score = (TP + TN) / (TP + FP + TN + FN)
        self.precision_score = TP / (TP + FP)
        self.recall_score = TP / (TP + FN)  # revocação
        self.f1_score = (2 * self.precision_score * self.recall_score) / (self.precision_score + self.recall_score)
