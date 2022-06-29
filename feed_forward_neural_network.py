#import tensorflow as tf
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt
#from tensorflow import keras
#from tensorflow.keras import Model
#from plot_keras_history import show_history, plot_history
#import matplotlib.pyplot as plt
#import statistics
#from numpy import sqrt 
#from scipy.stats import sem

#For jupyter notebook uncomment next line
#%matplotlib inline

# read csv input data
df = pd.read_csv('preprocessed_data.csv')
#print(df)

# Drop first column of dataframe
df = df.iloc[: , 1:]
#print(df)
dataTypeDict = dict(df.dtypes)

#print(dataTypeDict)



df["spam"]=df["spam"].astype('category').cat.codes

train, test = train_test_split(df, test_size=0.2)

print(train)