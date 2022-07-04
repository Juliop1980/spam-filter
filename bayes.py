import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# read csv input data
df = pd.read_csv('preprocessed_data.csv')

# Drop first column of dataframe
df = df.iloc[:, 1:]
dataTypeDict = dict(df.dtypes)

X = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
df["spam"] = df["spam"].astype('category').cat.codes

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

training_set = pd.DataFrame({'X': X_train, 'Y': Y_train})


# Vectorization model : Count vectorizer
vectorizer = CountVectorizer()
count_vectorizer = vectorizer.fit_transform(training_set['X'].values)
# init Multinomial naive bayes for the count vectorizer
classifier = MultinomialNB()
# training the model
classifier.fit(count_vectorizer, Y_train)
test_count_vectorizer = vectorizer.transform(X_test)
predictions_count_vect = classifier.predict(test_count_vectorizer)
# testing the accuracy of the model
accuracy_count_vect = accuracy_score(Y_test, predictions_count_vect)
print("Accuracy score of naïve Bayes with count vectorizer : " + str(accuracy_count_vect))


# Vectorization model : tf-idf
tfidf_vectorizer = TfidfVectorizer()
count_tfidf = tfidf_vectorizer.fit_transform(training_set['X'].values)
# init Multinomial naive bayes for tf-idf
classifier_tfidf = MultinomialNB()
# training the model
classifier_tfidf.fit(count_tfidf, Y_train)
test_count_tfidf = tfidf_vectorizer.transform(X_test)
predictions_tf = classifier_tfidf.predict(test_count_tfidf)
# testing the accuracy of the model
accuracy_count_tfidf = accuracy_score(Y_test, predictions_tf)
print("Accuracy score of naïve Bayes with tf-idf : " + str(accuracy_count_tfidf))


# vectorization model : countvectorizer - bag of 2-gram
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X = bigram_vectorizer.fit_transform(training_set['X'])
# init Multinomial naive bayes for bag of 2-gram
classifier_count_2 = MultinomialNB()
# training the model
classifier_count_2.fit(X, Y_train)
test_count_2 = bigram_vectorizer.transform(X_test)
predictions_2 = classifier_count_2.predict(test_count_2)
# testing the accuracy of the model
accuracy_count_2 = accuracy_score(Y_test, predictions_2)
print("Accuracy score of naïve Bayes with 2-gram vectorizer : " + str(accuracy_count_2))
