import pandas as pd
import sklearn
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# in order and same size.
# X mails, Y labels for the training set
X = []
Y = []


def fill_arrays(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append(row[1])
            Y.append(row[2])


fill_arrays("preprocessed_data.csv")
# print(X)
# print(Y)
# useful or not ?
training_set = pd.DataFrame({'X': X, 'Y': Y})

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(training_set['X'].values)
# print(counts)

classifier = MultinomialNB()
targets = training_set['Y'].values
classifier.fit(counts, targets)

"""examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)"""


