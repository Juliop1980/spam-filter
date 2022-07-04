import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import nltk
import string
from num2words import num2words
from nltk.corpus import stopwords
from collections import Counter

from nltk.stem import WordNetLemmatizer 



# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

#function to split the characters of a word to split the punctuation string given by string.punctuation
def split(word):
    return [char for char in word]


list_of_lengths_of_spam_mail = []
list_of_lengths_of_no_spam_mail = []

data_set_frame = pd.read_csv('Spam_Emails/Spam_Emails.csv', quotechar='"')

list_of_words_of_spam = []
list_of_words_of_no_spam = []

#function that gets every row from the dataframe and preprocess it
def preprocess(df_line):
    # This are global variables that we will use to get the insights of the data later and we take advantage of this function to do it while preprocessing
    global list_of_lengths_of_spam_mail
    global list_of_lengths_of_no_spam_mail
    global list_of_words_of_spam
    global list_of_words_of_no_spam

    # This variable will take the splitted cleaned text of every row
    aux_text_no_trash = []
    for i in (df_line['text'].lower()).split(" "):

        if i not in ( ["subject:", "subject", "'", '"', "_", "/", "-", "", " "] + stopwords.words('english') + split(string.punctuation)) and not i.isdigit() and len(i)>1:
            # We finally lemmatize the word

            aux_text_no_trash.append(lemmatizer.lemmatize(i))

        if i.isdigit():
            aux_text_no_trash.append(num2words(i))
    
    # Depending on the status of the spam field we add our meassurements and word to the correct list
    if df_line['spam'] == 1:
        list_of_words_of_spam += aux_text_no_trash
        list_of_lengths_of_spam_mail.append(len(aux_text_no_trash))
    
    if df_line['spam'] == 0:
        list_of_words_of_no_spam += aux_text_no_trash
        list_of_lengths_of_no_spam_mail.append(len(aux_text_no_trash))
        

    return " ".join(aux_text_no_trash)


# We call the function to preprocess the data and 
data_set_frame['text'] = data_set_frame.apply(preprocess, axis=1)

# we then add all the words and lengths to have an overall view of the data
list_of_all_words = list_of_words_of_spam + list_of_words_of_no_spam

list_of_lengths_all = list_of_lengths_of_spam_mail + list_of_lengths_of_no_spam_mail

#We prepare the data for plotting
data = [list_of_lengths_of_spam_mail, list_of_lengths_of_no_spam_mail]

# Making a box plot to show the distribution of lengths
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=0)

colors = ['#0000FF', '#00FF00']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linewidth=1.5,
                linestyle=":")

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B',
            linewidth=2)

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color='red',
               linewidth=3)

# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker='D',
              color='#e7298a',
              alpha=0.5)

# x-axis labels
ax.set_yticklabels(['Spam Emails', 'Normal Emails'])

# Adding title
plt.title("Comparison of length between spam and non spam emails")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(np.arange(0, max(list_of_lengths_all) + 1, 250))
plt.xlabel("Number of words in email")

# Uncomment next line if you want to save the plot
#plt.savefig("insights_of_data/Comparison_length_types_emails.png")

#Get number of normal emails and number of non spam emails
print("number of spam emails in data : "+ str(data_set_frame['spam'].value_counts()[1]))
print("number of non spam emails in data : "+ str(data_set_frame['spam'].value_counts()[0]))


# get unique words in both types of email

d = collections.defaultdict(int)
for x in list_of_words_of_spam: d[x] += 1
results = [x for x in list_of_words_of_spam if d[x] == 1]
print("Number of unique words in spam emails: " + str(len(results)))


d = collections.defaultdict(int)
for x in list_of_words_of_no_spam: d[x] += 1
results = [x for x in list_of_words_of_no_spam if d[x] == 1]
print("Number of unique words in non spam emails: " + str(len(results)))

d = collections.defaultdict(int)
for x in list_of_all_words: d[x] += 1
results = [x for x in list_of_all_words if d[x] == 1]
print("Number of unique words across all emails: " + str(len(results)))

c = Counter(list_of_words_of_spam)

print("Most common words in spam emails (<word>,<number_occurrences>): " + str(c.most_common(20)))

c = Counter(list_of_words_of_no_spam)

print("Most common words in non spam emails (<word>,<number_occurrences>): " + str(c.most_common(20)))


#Uncomment next line if you want to save preprocessed data to a csv
#data_set_frame.to_csv('preprocessed_data.csv')