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

for name, i_model in models:

    i_model_data = {}
    i_model.random_state = 100
    i_model_data["Name"] = name

    start = time.time()
    i_model.fit(train_x,train_y)
    end = time.time()

    i_model_data["Train_Time"] = end - start
    i_model_data["Train_R2_Score"] = metrics.r2_score(train_y,i_model.predict(train_x))
    i_model_data["Test_R2_Score"] = metrics.r2_score(test_y,i_model.predict(test_x))
    i_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,i_model.predict(test_x)))
    
    model_data.append(i_model_data)


model_scores = pd.DataFrame(model_data)
print(model_scores)