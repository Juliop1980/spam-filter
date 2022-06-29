import sklearn
import csv
from sklearn.feature_extraction.text import CountVectorizer

# ordonnés et de même taille.
# X mails, Y labels pour le training set
X = []
Y = []


def fill_les_tableaux(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append(row[1])
            Y.append(row[2])


fill_les_tableaux("preprocessed_data.csv")
# print(X)
# print(Y)

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(training_set['X'].values)