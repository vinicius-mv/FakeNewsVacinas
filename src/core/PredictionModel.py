import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class PredictionModel(object):
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.fake_tweets_train = None
        self.total_tweets_train = None
        self.model_train_accuracy = None
        self.model_test_accuracy = None

    # def preprocess_text(self, text_data):
    #     treated_text = []
    #     for setence in tqdm(text_data):
    #         setence = re.sub(r"\?+", "\?", setence)
    #         setence = re.sub(r'[^\w\s"?]', "", setence)
    #         # setence = re.sub(r"[^\w\s]", "", setence)
    #         treated_text.append(
    #             " ".join(
    #                 token.lower()
    #                 for token in str(setence).split()
    #                 if token not in stopwords.words("portuguese")
    #             )
    #         )
    #     return treated_text

    @staticmethod
    def preprocess_text(text):
        # Remove multiples questions marks
        text = re.sub(r"\?+", "?", text)

        # Tokenize text
        tokens = word_tokenize(text, language='portuguese')

        # Remove stopwords
        stop_words = set(stopwords.words('portuguese'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        preprocessed_text = ' '.join([lemmatizer.lemmatize(token) for token in filtered_tokens])

        return preprocessed_text

    @staticmethod
    def get_fake_tweets_length(Y: list):
        total = 0
        y: float
        for y in Y:
            if (float(y) == 1.0):
                total = total + 1.0
        return total

    def train(self, x, y, test_size=0.25):
        # Preprocess tweets
        x_preprocessed = [self.preprocess_text(tweet) for tweet in x]

        # Split data between train and test
        x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=test_size)

        # Set train size control fields
        self.total_tweets_train = len(x)
        self.fake_tweets_train = self.get_fake_tweets_length(y_train)

        # Create TF-IDF vectorizer
        # self.vectorizer = TfidfVectorizer()
        self.vectorizer = TfidfVectorizer(strip_accents="ascii", ngram_range=(1, 2))
        x_tfidf = self.vectorizer.fit_transform(x_train)
        
        # Train model
        self.model.fit(x_tfidf, y_train)

        # Calculate model accuracy
        self.model_train_accuracy = self.model_score(x_train, y_train)
        self.model_test_accuracy = self.model_score(x_test, y_test)

    def show_info(self):
        print('Model train - total tweets: ' + str(self.total_tweets_train))
        print('Model train - fake tweets: ' + str(self.fake_tweets_train))
        print('Model train - Accuracy: ' + str(self.model_train_accuracy))
        print('Model test - Accuracy: ' + str(self.model_test_accuracy))

    def model_score(self, x, y):
        x_tfidf = self.vectorizer.transform(x)
        return self.model.score(x_tfidf, y)

    def predict(self, tweets):
        # Preprocess tweets
        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]

        # Vectorize tweets
        x_tfidf = self.vectorizer.transform(preprocessed_tweets)

        # Make predictions
        predictions = self.model.predict(x_tfidf)

        return predictions