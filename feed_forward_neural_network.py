#import tensorflow as tf
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras import layers
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

emails_list = df['text'].tolist()


vectorizer = CountVectorizer(min_df=0, lowercase=False)
#X = vectorizer.fit_transform(emails_list)
vectorizer.fit(emails_list)
#vectorizer.fit(sentences)
vectorizer.vocabulary_
#vectorizer.get_feature_names_out()
#first_array = (X.toarray())[0]
#print(vectorizer.transform(emails_list).toarray())


#print(vectorizer.vocabulary_)
#print(vectorizer.transform(emails_list).toarray())

#print(train)
emails_list = df['text'].values
y= df['spam'].values

emails_train, emails_test, y_train, y_test = train_test_split(emails_list, y, test_size=0.2, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(emails_train)
X_train = vectorizer.transform(emails_train)
X_test  = vectorizer.transform(emails_test)
#print(X_train)

#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)
#score = classifier.score(X_test, y_test)
#print("Accuracy:", score)

input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())