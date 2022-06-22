import pandas as pd

df = pd.read_csv('Spam_Emails/Spam_Emails.csv', quotechar='"')

first_row = df.head(1)
print("First row Of Dataframe: ")
print(first_row)

#print(df.to_string()) 