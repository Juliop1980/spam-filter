import pandas as pd
#from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('WebAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#plt.use('TkAgg')


#list_of_lengths_all = []
list_of_lengths_of_spam_mail = []
list_of_lenghts_of_no_spam_mail=[]
data_set_frame = pd.read_csv('Spam_Emails/Spam_Emails.csv', quotechar='"')

data_frame_for_spam=data_set_frame.query("spam == 1")
data_frame_for_no_spam=data_set_frame.query("spam == 0")
#first_row = data_frame_for_no_spam.head(1)

#print("First row Of Dataframe: ")
#print(first_row)

#list_of_all_words=[]
list_of_words_of_spam= []
list_of_words_of_no_spam= []
#data_set_frame= data_set_frame.reset_index()
for index, row in data_frame_for_spam.iterrows():
    #print(row)
    text = row['text']
    list_of_words_of_spam.append(text.split(" "))
    
    #print(text)
    #print("-------------------------------------------------------------------------------")
    list_of_lengths_of_spam_mail.append(len(text))
    
    #if (len(text)) <30:
    #    print(text)
   
#print(df.to_string()) 

for index, row in data_frame_for_no_spam.iterrows():
    #print(row)
    text = row['text']
    list_of_words_of_no_spam.append(text.split(" "))
    
    #print(text)
    #print("-------------------------------------------------------------------------------")
    list_of_lenghts_of_no_spam_mail.append(len(text))

list_of_all_words = list_of_words_of_spam + list_of_words_of_no_spam
list_of_lengths_all = list_of_lengths_of_spam_mail + list_of_lenghts_of_no_spam_mail
#print(list_of_words_of_spam)
#print(list_of_words_of_no_spam)

    #print(words_in_text)

#print(list_of_lengths_of_spam_mail)
#print(list_of_lenghts_of_no_spam_mail)

#print(list_of_lengths_of_spam_mail)
#print(max(list_of_lengths_of_spam_mail))
#print(min(list_of_lengths_of_spam_mail))

#plt.plot(list_of_lengths_of_spam_mail)
#plt.hist(list_of_lengths_of_spam_mail, color = 'blue', edgecolor = 'black', bins = 20)
#plt.show()
""""
data = [list_of_lengths_of_spam_mail, list_of_lenghts_of_no_spam_mail]
plt.boxplot(data)
plt.yticks(np.arange(0, max(list_of_lengths_all)+1, 1800))
plt.title("Comparison of length between spam and nonspam emails")
plt.set_yticklabels(['Spam Emails', 'Non Spam Emails'])
plt.savefig("matplotlib.png") 
"""


# Creating dataset
np.random.seed(10)
data_1 = list_of_lengths_of_spam_mail
data_2 = list_of_lenghts_of_no_spam_mail

data = [data_1, data_2]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0)
 
colors = ['#0000FF', '#00FF00']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels(['Spam Emails', 'Normal Emails'])
 
# Adding title
plt.title("Comparison of length between spam and non spam emails")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(np.arange(0, max(list_of_lengths_all)+1, 3000))
plt.xlabel("Number of characters in email")
# show plot
plt.savefig("Comparison_length_types_emails.png")

