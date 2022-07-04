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

count_2gram = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
df["spam"] = df["spam"].astype('category').cat.codes

X_train, X_test, Y_train, Y_test = train_test_split(count_2gram, Y, test_size=0.2, random_state=0)

# scaling input data
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(Y_train)
# X_test = sc_X.fit_transform(X_test)

training_set = pd.DataFrame({'X': X_train, 'Y': Y_train})

# Vectorization model : Count vectorizer
# init vectorizer
vectorizer = CountVectorizer()
count_vectorizer = vectorizer.fit_transform(training_set['X'].values)

# training the model
# init Multinomial naive bayes
classifier_count = MultinomialNB()
classifier_count.fit(count_vectorizer, Y_train)

test_count_vectorizer = vectorizer.transform(X_test)
predictions_1 = classifier_count.predict(test_count_vectorizer)
# testing the accuracy of the model
accuracy_count_vect = accuracy_score(Y_test, predictions_1)
print("Accuracy score of count vectorizer : ", accuracy_count_vect)

# Vectorization model : tf-idf
bigram_vectorizer = TfidfVectorizer()
count_tfidf = bigram_vectorizer.fit_transform(training_set['X'].values)
# training the model
classifier_tfidf = MultinomialNB()
classifier_tfidf.fit(count_tfidf, Y_train)
# testing the accuracy of the model
test_count_tfidf = bigram_vectorizer.transform(X_test)
predictions_tf = classifier_tfidf.predict(test_count_tfidf)

accuracy_count_tfidf = accuracy_score(Y_test, predictions_tf)
print("Accuracy score of tf-idf : ", accuracy_count_tfidf)

# vectorization model : countvectorizer - 2-gram
bigram_vectorizer_2 = CountVectorizer(ngram_range=(2, 2))
count_2gram = bigram_vectorizer_2.fit_transform(training_set['X'])
# training the model
classifier_count_2 = MultinomialNB()

classifier_count_2.fit(count_2gram, Y_train)
# testing the accuracy of the model

test_count_2 = bigram_vectorizer_2.transform(X_test)
predictions_2 = classifier_count_2.predict(test_count_2)
accuracy_count_2 = accuracy_score(Y_test, predictions_2)
print("Accuracy score of 2-gram : ", accuracy_count_2)