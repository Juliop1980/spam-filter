from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import random
import gensim.downloader

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import sys

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

df = pd.read_csv('preprocessed_data.csv')

emails_list = df['text'].tolist()
#word_list_model = []
word_list = [word for line in emails_list for word in line.split()]
sentences = []

for sentence in emails_list:
    sentences.append(sentence.split())


random_words = random.sample(word_list, 20)

print("random words chosen: ")
print(random_words)
print("--------------------------------------------------------------------------------")
model = Word2Vec(window=3,min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
#word_list_model.append(random_words)
#print(word_list_model)
#print(random.sample(word_list, 20))
#model = Word2Vec(sentences=common_texts, vector_size=100, window=3, min_count=1, workers=4)
#model.save("word2vec.model")
#print(model)
#model = Word2Vec.load("word2vec.model")
#model.train([word_list_model], total_examples=1, epochs=1)

print("Most similar words for every random word")
for i in random_words:
        print("Most similar word for " + i +": ")
        #vector = model.wv[j]  # get numpy vector of a word
        #sims = model.wv.most_similar(j, topn=10)  # get other similar words
        print(model.wv.most_similar(i))
