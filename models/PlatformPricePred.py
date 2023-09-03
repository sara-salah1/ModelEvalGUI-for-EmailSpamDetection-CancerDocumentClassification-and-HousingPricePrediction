import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class HousePricePrediction:
    def __init__(self):
        self.data = None
        self.read_data()
        self.data_cleaning()
        self.numerical_feat = ['MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'TotalBsmtSF', 'SalePrice']
        for self.n in self.numerical_feat:
            self.detect_outliers(self.n)

        self.validation()
        self.Encoding()
        self.Normalization()
        self.EDA()
        self.split_data()
        self.train_model()
        self.save_model()
        self.accuracy()
        self.plot_rmse()

    def read_data(self):
        self.data = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/HousePricePrediction.xlsx - Sheet1.csv")
        print(self.data.shape)
        print(self.data.dtypes)

    def data_cleaning(self):
        print(self.data.isnull().sum())
        num_feat = ['TotalBsmtSF', 'BsmtFinSF2', 'SalePrice']
        cat_feat = ['MSZoning', 'Exterior1st']

        for n in num_feat:
            n_imputer = SimpleImputer(strategy='mean')
            self.data[[n]] = n_imputer.fit_transform(self.data[[n]])

        for c in cat_feat:
            c_imputer = SimpleImputer(strategy='most_frequent')
            self.data[[c]] = c_imputer.fit_transform(self.data[[c]])

        print(self.data.isnull().sum())
        self.data.drop_duplicates(keep='first')

    def detect_outliers(self, column_name):
        self.q1 = np.percentile(self.data[column_name], 25)
        self.q3 = np.percentile(self.data[column_name], 75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - (1.5 * self.iqr)
        self.upper_bound = self.q3 + (1.5 * self.iqr)
        self.outliers = [x for x in self.data[column_name] if x < self.lower_bound or x > self.upper_bound]
        if self.outliers:
            self.data[column_name] = np.where(self.data[column_name] < self.lower_bound, self.lower_bound,
                                              self.data[column_name])
            self.data[column_name] = np.where(self.data[column_name] > self.upper_bound, self.upper_bound,
                                              self.data[column_name])
        print("lower bound:", self.lower_bound)
        print("upper bound:", self.upper_bound)
        print(f"Outliers for column {column_name}:\n{self.outliers}")
        print("--------------------------------------------------------------------------")

    def validation(self):
        self.data = self.data[(self.data['BsmtFinSF2'] >= 0)]
        self.data = self.data[(self.data['TotalBsmtSF'] >= 0)]

        self.data.Exterior1st.value_counts()

        # check validation for YearRemodAdd
        plt.hist(self.data['YearRemodAdd'], bins=20, edgecolor='k')
        plt.xlabel('Year Remodeled')
        plt.ylabel('Frequency')
        plt.title('Distribution of Remodeling Years')
        plt.show()

        # check validation for YearBuilt
        plt.hist(self.data['YearBuilt'], bins=20, edgecolor='k')
        plt.xlabel('Year Built')
        plt.ylabel('Frequency')
        plt.title('Distribution of Remodeling Years')
        plt.show()

        self.data = self.data[(self.data['OverallCond'] >= 1) & (self.data['OverallCond'] <= 10)]

        print(self.data.BldgType.value_counts())
        print(self.data.LotConfig.value_counts())

        self.data = self.data[self.data['LotArea'] > 0]

        print(self.data.MSZoning.value_counts())

        self.li = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]

        self.mask = self.data['MSSubClass'].isin(self.li)
        self.data = self.data[self.mask]
        self.data = self.data.drop(columns='Id')

        print(self.data.head())

    def Encoding(self):
        categ_feat = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
        for feat in categ_feat:
            encoder = LabelEncoder()
            self.data[feat] = encoder.fit_transform(self.data[feat])
        print(self.data.head())

    def Normalization(self):
        for n in self.numerical_feat:
            scaler = StandardScaler()
            self.data[[n]] = scaler.fit_transform(self.data[[n]])
        print(self.data.head())

    def EDA(self):
        print(self.data.MSZoning.value_counts())

        plt.figure(figsize=(5, 5))
        sns.countplot(y='BldgType', data=self.data)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.countplot(y='MSZoning', data=self.data)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.countplot(y='LotConfig', data=self.data)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.countplot(y='Exterior1st', data=self.data)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.distplot(self.data.BsmtFinSF2)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.distplot(self.data.MSSubClass)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.distplot(self.data.TotalBsmtSF)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.distplot(self.data.LotArea)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.distplot(self.data.OverallCond)
        plt.show()

        plt.figure(figsize=(8, 8))
        sns.heatmap(self.data.corr(), annot=True)
        plt.show()

        plt.figure(figsize=(5, 5))
        sns.pairplot(self.data)
        plt.show()

    def split_data(self):
        self.feature = self.data.drop(columns='SalePrice')
        self.target = self.data['SalePrice']

        self.x_trainHPP, self.x_testHPP, self.y_trainHPP, self.y_testHPP = train_test_split(self.feature, self.target,
                                                                                            test_size=0.2,
                                                                                            random_state=100)

        self.x_testHPP.to_csv('x_testHPP.csv', index=False)
        self.y_testHPP.to_csv('y_testHPP.csv', index=False)

        print("Feature training dataset shape: ", self.x_trainHPP.shape)
        print(self.x_trainHPP.columns)
        print(self.x_testHPP.columns)
        print(self.x_testHPP.head())
        print("Label training dataset shape: ", self.y_trainHPP.shape)
        print("Feature testing dataset shape: ", self.x_testHPP.shape)
        print("Label testing dataset shape: ", self.y_testHPP.shape)

    def train_model(self):
        self.LR_model = LinearRegression()
        self.LR_model.fit(self.x_trainHPP, self.y_trainHPP)

    def save_model(self):
        joblib.dump(self.LR_model, 'C:/Users/LEGION/PycharmProjects/GradProjectITI/LR_model.pk1')

    def accuracy(self):
        predicted = self.LR_model.predict(self.x_testHPP)
        self.rmse = np.sqrt(mean_squared_error(self.y_testHPP, predicted))
        print("Root Mean Square : ", self.rmse * 100)

    def plot_rmse(self):
        model_names = ['Linear Regression']
        rmse_values = [self.rmse]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(model_names, rmse_values, color='#40826d')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error (RMSE) for House Price Prediction')
        plt.ylim(0, max(rmse_values) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{self.rmse * 100:.2f}%', ha='center',
                     color='black', fontsize=12)

        plt.show()
