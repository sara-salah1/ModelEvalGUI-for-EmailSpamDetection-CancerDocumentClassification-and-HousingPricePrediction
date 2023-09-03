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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class EmailSpamDetection:
    def __init__(self):
        self.data = None
        self.encoder = None
        self.feature_extraction = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.knn_model, self.dt_model, self.nb_model = None, None, None
        self.train_accuracy = None
        self.train_confusion_matrix = None
        self.test_accuracy = None
        self.test_confusion_matrix = None
        self.EmailDetect()

    def EmailDetect(self):
        self.read_data()
        self.data_cleaning()
        self.data_transformation()
        self.data['edited_text'] = self.data['Email'].apply(self.email_preprocessing)
        print(self.data['edited_text'].head())
        self.vectorization()
        self.split_data()
        self.train_models()
        self.save_models()
        self.train_accuracy = self.get_train_accuracy()
        self.train_confusion_matrix = self.get_train_confusion_metrix()
        self.test_accuracy = self.get_test_accuracy()
        self.test_confusion_matrix = self.get_test_confusion_metrix()
        print("train_accuracy", self.train_accuracy)
        print("train_confusion_matrix", self.train_confusion_matrix)
        print("test_accuracy", self.test_accuracy)
        print("test_confusion_matrix", self.test_confusion_matrix)

    def read_data(self):
        self.data = pd.read_csv('C:/Users/LEGION/PycharmProjects/GradProjectITI/spam.csv', encoding='ISO-8859-1')
        print(self.data.head())
        print("data Info", self.data.info())
        print(self.data.describe())
        print(self.data.value_counts())

    def data_cleaning(self):
        self.data = self.data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
        self.data.rename(columns={'v1': 'Target', 'v2': 'Email'}, inplace=True)
        print("Null values: ", self.data.isnull().sum())

        self.encoder = LabelEncoder()
        self.data['Target'] = self.encoder.fit_transform(self.data['Target'])
        print(self.data.head())
        print("Check Duplicate Values ", self.data.drop_duplicates(keep='first'))
        print(self.data['Target'].value_counts())

    def data_transformation(self):
        self.data['no_char'] = self.data['Email'].apply(len)
        self.data['no_word'] = self.data['Email'].apply(lambda x: len(str(x).split()))
        self.data['no_sentence'] = self.data['Email'].apply(lambda x: len(x.split(',')))

        print(self.data.head())

        plt.figure(figsize=(6, 6))
        sns.pairplot(self.data, hue='Target')
        plt.show()

        self.data = self.data.drop(columns='no_char')
        print("data after drop no_char column", self.data.head())

    def email_preprocessing(self, text):
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
        self.feature_extraction = TfidfVectorizer(min_df=1)
        self.x = (self.feature_extraction.fit_transform(self.data['edited_text']))
        self.y = self.data['Target']

        print("feature data shape(no of rows, no of cols)", self.x.shape)
        print("print feature data itself", self.x)

        self.x = self.x.toarray()

        print("label data shape(no of rows, no of cols): ", self.y.shape)
        print("label data:", self.y.head())

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=100)
        # Save x_test and y_test as CSV files
        np.savetxt('x_test.csv', self.x_test, delimiter=',')
        np.savetxt('y_test.csv', self.y_test, delimiter=',')
        print("Feature training dataset shape: ", self.x_train.shape)
        print("Label training dataset shape: ", self.y_train.shape)
        print("Feature testing dataset shape: ", self.x_test.shape)
        print("Label testing dataset shape: ", self.y_test.shape)

    def train_models(self):
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.knn_model.fit(self.x_train, self.y_train)

        self.dt_model = DecisionTreeClassifier()
        self.dt_model.fit(self.x_train, self.y_train)

        self.nb_model = MultinomialNB()
        self.nb_model.fit(self.x_train, self.y_train)

    def save_models(self):
        joblib.dump(self.knn_model, 'C:/Users/LEGION/PycharmProjects/GradProjectITI/KNN_model.pk1')
        joblib.dump(self.dt_model, 'C:/Users/LEGION/PycharmProjects/GradProjectITI/dt_model.pk1')
        joblib.dump(self.nb_model, 'C:/Users/LEGION/PycharmProjects/GradProjectITI/naive_model.pk1')

    def get_train_accuracy(self):
        y_pred_knn = self.knn_model.predict(self.x_train)
        y_pred_dt = self.dt_model.predict(self.x_train)
        y_pred_nb = self.nb_model.predict(self.x_train)

        accuracy_knn = accuracy_score(self.y_train, y_pred_knn)
        accuracy_dt = accuracy_score(self.y_train, y_pred_dt)
        accuracy_nb = accuracy_score(self.y_train, y_pred_nb)

        return {
            'knn_accuracy': accuracy_knn,
            'dt_accuracy': accuracy_dt,
            'nb_accuracy': accuracy_nb,
        }

    def get_train_confusion_metrix(self):
        y_pred_knn = self.knn_model.predict(self.x_train)
        y_pred_dt = self.dt_model.predict(self.x_train)
        y_pred_nb = self.nb_model.predict(self.x_train)

        confusion_matrix_knn = confusion_matrix(self.y_train, y_pred_knn)
        confusion_matrix_dt = confusion_matrix(self.y_train, y_pred_dt)
        confusion_matrix_nb = confusion_matrix(self.y_train, y_pred_nb)

        return {
            'knn_confusion_matrix': confusion_matrix_knn,
            'dt_confusion_matrix': confusion_matrix_dt,
            'nb_confusion_matrix': confusion_matrix_nb
        }

    def get_test_accuracy(self):
        y_pred_knn = self.knn_model.predict(self.x_test)
        y_pred_dt = self.dt_model.predict(self.x_test)
        y_pred_nb = self.nb_model.predict(self.x_test)

        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        accuracy_nb = accuracy_score(self.y_test, y_pred_nb)

        return {
            'knn_accuracy': accuracy_knn,
            'dt_accuracy': accuracy_dt,
            'nb_accuracy': accuracy_nb,
        }

    def get_test_confusion_metrix(self):
        y_pred_knn = self.knn_model.predict(self.x_test)
        y_pred_dt = self.dt_model.predict(self.x_test)
        y_pred_nb = self.nb_model.predict(self.x_test)

        confusion_matrix_knn = confusion_matrix(self.y_test, y_pred_knn)
        confusion_matrix_dt = confusion_matrix(self.y_test, y_pred_dt)
        confusion_matrix_nb = confusion_matrix(self.y_test, y_pred_nb)

        return {
            'knn_confusion_matrix': confusion_matrix_knn,
            'dt_confusion_matrix': confusion_matrix_dt,
            'nb_confusion_matrix': confusion_matrix_nb
        }

    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
