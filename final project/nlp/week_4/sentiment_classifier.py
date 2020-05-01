from sklearn.externals import joblib
import string
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./models/DefaultLogisticBigramUnprocessedTextSentiment.pkl")
        self.vectorizer = joblib.load("./models/BigramUnprocessedVectorizer.pkl")
        self.classes_dict = {0: "негативный", 1: "позитивный", -1: "putin error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.75:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""
    
    def delete_punctuation(self,line):
        translation_table = str.maketrans('', '', string.punctuation)
        return line.translate(translation_table).lower()


    def format_str(self,text):
        word_list = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in word_list]
        stroka = self.delete_punctuation(' '.join(words))
        words = [word.strip() for word in stroka.split()]
        return [' '.join(words)]

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform(self.format_str(text))
            return (self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max())
        except:
            return -1, 0.8

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]


