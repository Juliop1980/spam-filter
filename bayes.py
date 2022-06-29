import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# read csv input data
df = pd.read_csv('preprocessed_data.csv')

# Drop first column of dataframe
df = df.iloc[:, 1:]
dataTypeDict = dict(df.dtypes)

X = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
# print(X)
# print(Y)
df["spam"] = df["spam"].astype('category').cat.codes

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
"""print(X_train)
print(Y_train)
print(X_test)
print(Y_test)
train, test = train_test_split(df, test_size=0.2)
X_train = train.iloc[:, 0]

Y_train = train.iloc[:, 1]
# print(X)
# print(Y)
X_test = test.iloc[:, 0]
Y_test = test.iloc[:, 1]
"""

# scaling input data
sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)

training_set = pd.DataFrame({'X': X_train, 'Y': Y_train})

vectorizer = CountVectorizer()
count_vectorizer = vectorizer.fit_transform(training_set['X'].values)

bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
count_tfidf = bigram_vectorizer.fit_transform(training_set['X'].values)

# init Multinomial naive bayes
classifier = MultinomialNB()

# training the model
classifier.fit(count_vectorizer, Y_train)

test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(test_counts)
# print(predictions)
# testing the model



accuracy_count_vect = accuracy_score(Y_test, predictions)
print(accuracy_count_vect)
