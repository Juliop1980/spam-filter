import pandas as pd
import nltk
import string
from num2words import num2words
from statistics import mean

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('stopwords')

ps = PorterStemmer()


def split(word):
    return [char for char in word]


list_of_lengths_of_spam_mail = []
list_of_lengths_of_no_spam_mail = []
data_set_frame = pd.read_csv('Spam_Emails/Spam_Emails.csv', quotechar='"')

data_frame_for_spam = data_set_frame.query("spam == 1")
data_frame_for_no_spam = data_set_frame.query("spam == 0")

list_of_words_of_spam = []
list_of_words_of_no_spam = []
for index, row in data_frame_for_spam.iterrows():
    # print(row)
    text = row['text'].lower()
    aux_text_no_trash = [ps.stem(i) for i in text.split(" ") if i not in (
            ["subject:", "subject", "'", '"', "_", "/", "-", ""] + stopwords.words('english') + split(
                string.punctuation)) and not i.isdigit()]
    numbers_as_words = [num2words(i) for i in text.split(" ") if i.isdigit()]
    list_of_words_of_spam += (aux_text_no_trash + numbers_as_words)

    list_of_lengths_of_spam_mail.append(len(aux_text_no_trash))

for index, row in data_frame_for_no_spam.iterrows():
    text = row['text'].lower()
    aux_text_no_trash = [ps.stem(i) for i in text.split(" ") if i not in (
            ["subject:", "subject", "'", '"', "_", "/", "-", ""] + stopwords.words('english') + split(
                string.punctuation)) and not i.isdigit()]
    numbers_as_words = [num2words(i) for i in text.split(" ") if i.isdigit()]

    list_of_words_of_no_spam += (aux_text_no_trash + numbers_as_words)
    list_of_lengths_of_no_spam_mail.append(len(aux_text_no_trash))

list_of_all_words = list_of_words_of_spam + list_of_words_of_no_spam

list_of_lengths_all = list_of_lengths_of_spam_mail + list_of_lengths_of_no_spam_mail

avg_length_spam = mean(list_of_lengths_of_spam_mail)
avg_length_ham = mean(list_of_lengths_of_no_spam_mail)

print(avg_length_spam)
print(avg_length_ham)


data_1 = list_of_lengths_of_spam_mail
data_2 = list_of_lengths_of_no_spam_mail

for i in range(400):
    print(list_of_all_words[i])