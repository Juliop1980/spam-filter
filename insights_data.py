import pandas as pd
from matplotlib import pyplot as plt


list_of_lengths_all = []
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
#print(list_of_words_of_spam)
#print(list_of_words_of_no_spam)

    #print(words_in_text)

#print(list_of_lengths_of_spam_mail)
#print(list_of_lenghts_of_no_spam_mail)

plt.plot(list_of_lengths_of_spam_mail)
plt.show()
