#import tensorflow as tf
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras import layers
from keras.models import Sequential
from keras.backend import clear_session
from keras.layers import Dense, Softmax, Dropout
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
import sys


plt.style.use('ggplot')
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



def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# read csv input data
df = pd.read_csv('preprocessed_data.csv')
#print(df)

#vectorizer= TfidfVectorizer()
#tfidf_vect = vectorizer.fit_transform(df)
#tfidf_vect_df = pd.DataFrame(tfidf_vect.toarray(), index = df.index, columns=df.columns)
#print(tfidf_vect_df)

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
word_count_vector =vectorizer.fit(emails_train)

try:
    vectorization = sys.argv[1]
except:
    vectorization = ""

if vectorization =="count":
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
    #model.add(Dropout(0.2))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    history = model.fit(X_train, y_train, epochs=80,verbose=False,validation_data=(X_test, y_test),batch_size=100)

    clear_session()

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

    print("Training Accuracy: {:.4f}".format(accuracy))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)
    plt.savefig('Neural_Network_results/NNmodel_count_vectorizerresults_.png')

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
print(tfidf_transformer.fit(word_count_vector))


