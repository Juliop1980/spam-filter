import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras import layers
from keras.models import Sequential
from keras.backend import clear_session
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


plt.style.use('ggplot')


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


# Drop first column of dataframe
df = df.iloc[:, 1:]
dataTypeDict = dict(df.dtypes)

df["spam"] = df["spam"].astype('category').cat.codes

train, test = train_test_split(df, test_size=0.2)

emails_list = df['text'].tolist()

vectorizer = CountVectorizer(min_df=0, lowercase=False)

vectorizer.fit(emails_list)
emails_list = df['text'].values
y = df['spam'].values

emails_train, emails_test, y_train, y_test = train_test_split(emails_list, y, test_size=0.2, random_state=1000)

try:
    vectorization = sys.argv[1]
except:
    vectorization = ""

if vectorization == "count":
    vectorizer = CountVectorizer()
    word_count_vector = vectorizer.fit(emails_train)
    X_train = vectorizer.transform(emails_train)
    X_test = vectorizer.transform(emails_test)

if vectorization == "TF-IDF":
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(emails_train).toarray()
    X_test = vectorizer.transform(emails_test).toarray()

if vectorization =="2-gram":
    vectorizer = CountVectorizer(ngram_range = (2, 2))
    word_count_vector = vectorizer.fit(emails_train)
    X_train = vectorizer.transform(emails_train)
    X_test = vectorizer.transform(emails_test)

if vectorization == "3-gram":
    vectorizer = CountVectorizer(ngram_range = (3, 3))
    word_count_vector = vectorizer.fit(emails_train)
    X_train = vectorizer.transform(emails_train)
    X_test = vectorizer.transform(emails_test)

if vectorization =="4-gram":
    vectorizer = CountVectorizer(ngram_range = (4,4))
    word_count_vector =vectorizer.fit(emails_train)
    X_train = vectorizer.transform(emails_train)
    X_test  = vectorizer.transform(emails_test)

input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])

history = model.fit(X_train, y_train, epochs=80,verbose=False,validation_data=(X_test, y_test),batch_size=100)

clear_session()

loss, accuracy, f1_score, precision, recall = model.evaluate(X_train, y_train, verbose=False)

print(model.get_config())

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=False)

