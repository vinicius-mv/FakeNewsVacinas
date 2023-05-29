import re
import nltk
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#spacy.cli.download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")

nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

class PredictionModel(object):
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.fake_tweets_train = None
        self.total_tweets_train = None
        self.model_train_accuracy = None
        self.model_test_accuracy = None
        
    @staticmethod
    def preprocess_text(text):
        # Remove multiples questions mark
        text = re.sub(r"\?+", "?", text)
        
        # Filter symbols, punctuation, etc.
        text = re.sub(r"[^\w\s?\"]", "", text)
        
        # Convert abreviations
        text = re.sub(r"\sq\s", " que ", text)
        text = re.sub(r"\svc\s", " você ", text)
        text = re.sub(r"\smsm\s", " mesmo ", text)

        # Lemmatize tokens and remove stopwords 
        stop_words = set(stopwords.words('portuguese'))
        doc = nlp(text)
        tokens  = [ token.lemma_ for token in doc if token.text not in stop_words ]
        
        # create a string from tokens
        preprocessed_text = ' '.join(tokens)
        
        # remove multiple whitespaces
        preprocessed_text = re.sub("\s+", " ", preprocessed_text)
        
        return preprocessed_text

    @staticmethod
    def get_fake_tweets_length(Y: list):
        total = 0
        y: float
        for y in Y:
            if (float(y) == 1.0):
                total = total + 1.0
        return total

    def train(self, x, y, test_size, seed=19):
        # Preprocess tweets
        x_preprocessed = [self.preprocess_text(tweet) for tweet in x]

        # Split data between train and test
        x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=test_size, random_state=seed)

        # Set train size control fields
        self.total_tweets_train = len(x)
        self.fake_tweets_train = self.get_fake_tweets_length(y_train)

        # Create TF-IDF vectorizer
        # self.vectorizer = TfidfVectorizer()
        self.vectorizer = TfidfVectorizer(strip_accents="ascii", ngram_range=(1, 1))
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
