import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import read_ahu_one, MergeData

from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

import time

class Helper(object):

    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

# rf = Helper(clf=RandomForestRegressor, seed=0, params=None)
# nn = Helper(clf=MLPRegressor, seed=0, params=None)
# lr = Helper(clf=LinearRegression, seed=0, params=None)

evac, ext, hum, sup, rec = read_ahu_one()
data = MergeData()

train, test = train_test_split(data, test_size=0.2)

features = ['exh', 'ext', 'hum', 'rec']
target = ['sup']

sc = MinMaxScaler()

train_x = sc.fit_transform(train[features])
train_y = sc.fit_transform(train[target])
test_x = sc.fit_transform(test[features])
test_y = sc.fit_transform(test[target])

print("Your features are " + str(features))
print("Your target is " + str(target))

models = [
           ['RandomForest ',RandomForestRegressor()],
           ['Lasso ', Lasso()],
           ['Ridge ', Ridge()],
           ['MLPRegressor ', MLPRegressor(activation='relu', solver='adam', learning_rate='adaptive',
           max_iter=10000, learning_rate_init=0.001, alpha=0.01)]
         ]

model_data = []

for name,curr_model in models:

    curr_model_data = {}
    curr_model.random_state = 100
    curr_model_data["Name"] = name

    start = time.time()
    curr_model.fit(train_x,train_y)
    end = time.time()

    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_x))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_x))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_x)))
    
    model_data.append(curr_model_data)


model_scores = pd.DataFrame(model_data)
print(model_scores)