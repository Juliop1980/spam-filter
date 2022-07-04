from gensim.models import Word2Vec
import random
import gensim.downloader
import pandas as pd

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

df = pd.read_csv('preprocessed_data.csv')

emails_list = df['text'].tolist()
word_list = [word for line in emails_list for word in line.split()]
sentences = []

for sentence in emails_list:
    sentences.append(sentence.split())

random_words = random.sample(word_list, 20)

print("random words chosen: ")
print(random_words)
print("--------------------------------------------------------------------------------")
model = Word2Vec(window=3, min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors

print("Most similar words for every random word")
for i in random_words:
    print("Most similar word for " + i + ": ")
    print(model.wv.most_similar(i))
