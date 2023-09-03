import re
import emoji
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class CancerDocClass:
    def __init__(self):
        self.data = None
        self.encoder = None
        self.feature_extraction = None
        self.x_trainCDC, self.x_testCDC, self.y_trainCDC, self.y_testCDC = None, None, None, None
        self.kmean_model = None
        self.train_accuracy = None
        self.train_confusion_matrix = None
        self.test_accuracy = None
        self.test_confusion_matrix = None
        self.DocClassif()

    def DocClassif(self):
        self.read_data()
        self.data_cleaning()
        self.data['edited_text'] = self.data['Document'].apply(self.Doc_preprocessing)
        print(self.data['edited_text'].head())
        self.vectorization()
        self.feature_selection()
        self.split_data()
        self.train_models()
        self.save_models()
        self.train_confusion_matrix = self.get_train_confusion_metrix()
        self.test_confusion_matrix = self.get_test_confusion_metrix()
        print("train_confusion_matrix", self.train_confusion_matrix)
        print("test_confusion_matrix", self.test_confusion_matrix)


    def read_data(self):
        self.data = pd.read_csv('C:/Users/LEGION/PycharmProjects/GradProjectITI/CancerDoc.csv', encoding='ISO-8859-1')
        print(self.data.head())
        print("data Info", self.data.info())
        print(self.data.describe())
        print(self.data.value_counts())

    def data_cleaning(self):
        self.data = self.data.drop(columns='Unnamed: 0')
        self.data.rename(columns={'0': 'Target', 'a': 'Document'}, inplace=True)

        print(self.data.head())

        self.data['Target'].value_counts().plot(kind='bar')
        plt.xticks(rotation=45)
        plt.title('Types of Cancer')
        plt.tight_layout()
        plt.show()

        print("Null values: ", self.data.isnull().sum())

        self.encoder = LabelEncoder()
        self.data['Target'] = self.encoder.fit_transform(self.data['Target'])
        print(self.data.head())
        print("Check Duplicate Values ", self.data.drop_duplicates(keep='first'))
        print(self.data['Target'].value_counts())

    def Doc_preprocessing(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        tokens = [stemmer.stem(word) for word in words if word not in stop_words]
        processed_text = ' '.join(tokens)
        processed_text = emoji.demojize(processed_text)
        processed_text = (BeautifulSoup(processed_text, 'html.parser').getText())
        processed_text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', processed_text)

        return processed_text

    def vectorization(self):
        self.feature_extraction = TfidfVectorizer(min_df=5)
        self.x = (self.feature_extraction.fit_transform(self.data['edited_text']))
        self.y = self.data['Target']

        print("feature data shape(no of rows, no of cols)", self.x.shape)
        print("print feature data itself", self.x)

        self.x = self.x.toarray()

        print("label data shape(no of rows, no of cols): ", self.y.shape)
        print("label data:", self.y.head())

    def feature_selection(self):
        selector = SelectKBest(score_func=f_classif, k=100)
        self.x = selector.fit_transform(self.x, self.y)

    def split_data(self):
        self.x_trainCDC, self.x_testCDC, self.y_trainCDC, self.y_testCDC = train_test_split(self.x, self.y,
                                                                                            test_size=0.2,
                                                                                            random_state=100)
        np.savetxt('x_testCDC.csv', self.x_testCDC, delimiter=',')
        np.savetxt('y_testCDC.csv', self.y_testCDC, delimiter=',')
        print("Feature training dataset shape: ", self.x_trainCDC.shape)
        print("Label training dataset shape: ", self.y_trainCDC.shape)
        print("Feature testing dataset shape: ", self.x_testCDC.shape)
        print("Label testing dataset shape: ", self.y_testCDC.shape)

    def train_models(self):
        self.kmean_model = KMeans(n_clusters=5, random_state=42)
        self.kmean_model.fit(self.x_trainCDC, self.y_trainCDC)

    def save_models(self):
        joblib.dump(self.kmean_model, 'C:/Users/LEGION/PycharmProjects/GradProjectITI/Kmean_model.pk1')

    def get_train_confusion_metrix(self):
        y_pred_kmean = self.kmean_model.predict(self.x_trainCDC)
        confusion_matrix_kmean = confusion_matrix(self.y_trainCDC, y_pred_kmean)

        return {
            'kmean_confusion_matrix': confusion_matrix_kmean,
        }

    def get_test_confusion_metrix(self):
        y_pred_kmean = self.kmean_model.predict(self.x_testCDC)
        confusion_matrix_kmean = confusion_matrix(self.y_testCDC, y_pred_kmean)

        return {
            'kmean_confusion_matrix': confusion_matrix_kmean
        }

    def plot_confusion_matrix(self, matrix, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
