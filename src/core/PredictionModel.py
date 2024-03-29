import re
import nltk
import spacy
import pandas as pd
import seaborn as sn
import shap

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

        self.X_train = None
        self.X_train_tfidf = None
        self.X_test = None
        self.X_test_tfidf = None
        self.Y_train = None
        self.Y_test = None

        self.explainer = None

    @staticmethod
    def preprocess_text(text):
        # normalize message
        text = text.lower()

        # Remove multiples questions mark
        text = re.sub(r"\?+", "?", text)

        # Filter symbols, punctuation, etc.
        text = re.sub(r"[^\w\s?\"]", "", text)

        # Convert abbreviations
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

    def train(self, X, Y, test_size, seed=19):
        # Preprocess tweets
        X_preprocessed = [self.preprocess_text(tweet) for tweet in X]

        # Split data between train and test
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_preprocessed, Y, test_size=test_size, random_state=seed)

        # Set train size control fields
        self.total_tweets = len(X)
        self.total_fake_tweets = self.get_fake_tweets_length(Y)

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(strip_accents="ascii", ngram_range=(1, 1))
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)

        # Train model
        self.model.fit(self.X_train_tfidf, self.Y_train)

        # model scores
        self.calculate_scores(self.X_test, self.Y_test)

    def dataset_info(self):
        print('Total tweets: ' + str(self.total_tweets))
        print('Fake tweets: ' + str(self.total_fake_tweets))
        print(f'Test proportion: {len(self.X_test) / len(self.X_train)}')

    def model_info(self):
        print(f"Model {self.model_name}")
        print('Model - accuracy score: ' + str(self.accuracy_score))
        print('Model - precision score: ' + str(self.precision_score))
        print('Model - recall score: ' + str(self.recall_score))
        print('Model - f1 score: ' + str(self.f1_score))

    def predict(self, tweets):
        # Preprocess tweets
        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]

        # Vectorize tweets
        x_tfidf = self.vectorizer.transform(preprocessed_tweets)

        # Make predictions
        predictions = self.model.predict(x_tfidf)

        return predictions

    def calculate_scores(self, x_test, y_test):
        self.X_test_tfidf = self.vectorizer.transform(x_test)
        y_pred = self.model.predict(self.X_test_tfidf)

        self.crosstab = pd.crosstab(y_pred, y_test)
        TP = self.crosstab[1][1]  # true positives
        TN = self.crosstab[0][0]  # true negatives
        FP = self.crosstab[0][1]  # false positives
        FN = self.crosstab[1][0]  # false negatives
        self.accuracy_score = (TP + TN) / (TP + FP + TN + FN)
        self.precision_score = TP / (TP + FP)
        self.recall_score = TP / (TP + FN)  # revocação
        self.f1_score = (2 * self.precision_score * self.recall_score) / (self.precision_score + self.recall_score)

    def get_confusion_matrix(self):
        ax = sn.heatmap(self.crosstab, annot=True)
        ax.set_title(self.model_name)
        ax.set(xlabel="Valores Reais")
        ax.set(ylabel="Valores Previstos")
        plt.show()

    def explainer_summary_plot(self, tweets):

        self._load_explainer()

        # Explain predictions for a specific instance
        # Preprocess tweets
        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]

        # Vectorize tweets
        x_tfidf = self.vectorizer.transform(preprocessed_tweets)

        shap_values = self.explainer(x_tfidf)

        shap.summary_plot(shap_values, x_tfidf, plot_size=(6,6))

    def explainer_beeswarm_plot(self, tweets):

        self._load_explainer()

        # Explain predictions for a specific instance
        # Preprocess tweets
        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]

        # Vectorize tweets
        x_tfidf = self.vectorizer.transform(preprocessed_tweets)

        shap_values = self.explainer(x_tfidf)

        shap.plots.beeswarm(shap_values, plot_size=(6,6))



    def _pre_process_tweets(self, tweets):
        # Preprocess tweets
        preprocessed_tweets = []
        for tweet in tweets:
            preprocessed_tweets.append(self.preprocess_text(tweet))
        return preprocessed_tweets

    def _load_explainer(self):
        if (self.explainer is not None):
            return

        self.explainer = shap.Explainer(self.model, self.X_train_tfidf, feature_names=self.vectorizer.get_feature_names_out())
