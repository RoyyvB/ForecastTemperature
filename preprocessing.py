import pandas as pd
import numpy as np
import glob

import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

from functools import reduce

path = r"C:\Users\Roy\Desktop\ForecastTemperature"

def read_ahu_one():

    ahu_one = "\AHU1\\"

    exh = pd.read_csv(path + ahu_one + "ahu1_evac.csv")
    ext = pd.read_csv(path + ahu_one + "ahu1_ext.csv")
    hum = pd.read_csv(path + ahu_one + "ahu1_hum.csv")
    sup = pd.read_csv(path + ahu_one + "ahu1_in.csv")
    rec = pd.read_csv(path + ahu_one + "ahu1_rec.csv")

    exh.columns = ['date', 'exh'] # Exhaust temperature
    ext.columns = ['date', 'ext'] # Outdoor temperature
    hum.columns = ['date', 'hum'] # Humidity
    sup.columns = ['date', 'sup'] # Supply temperature
    rec.columns = ['date', 'rec'] # Recirculation temperature

    return exh, ext, hum, sup, rec

evac, ext, hum, sup, rec = read_ahu_one()

def MergeData():

    list_of_data = [evac, ext, hum, sup, rec]

    data = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='outer'), list_of_data)

    # Reorder columns.
    data = data[['date', 'exh', 'ext', 'hum', 'rec', 'sup']]

    return data

data = MergeData()

def PrepareData():

    evac, ext, hum, sup, rec = read_ahu_one()

    data = MergeData()

    return data

def GenerateProfile(data, title_name, title_output):

    profile = ProfileReport(data, title=title_name)
    profile.to_file(output_file=title_output)

def CorrelationMap():
    
    print(data.columns)

    DC = data.iloc[:, 1:5]
    CRLN = DC.corr()

    f, ax = plt.subplots(figsize=(17, 14))
    sns.heatmap(CRLN, annot=True, fmt=".2f")
    plt.xticks(range(len(CRLN.columns)), CRLN.columns)
    plt.yticks(range(len(CRLN.columns)), CRLN.columns)
    plt.show()
