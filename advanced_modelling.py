import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import PrepareData
from parameters import rf_params

from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

data = PrepareData()
