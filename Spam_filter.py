#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import os

#Reading the file into a pandas dataframe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
df = pd.read_csv(DATA_PATH)

#droping possible duplicates
df.drop_duplicates(inplace = True)

#converting the numeric- spam = 1 and ham = 0
df['Category'] = df['Category'].replace(['ham', 'spam'], ['Not spam', 'Spam'])

# defining x and y
x = df['Message']
y= df['Category']

#spliting our data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#converting the x_train into numeric using count vectorizer
cv= CountVectorizer(stop_words = 'english')
x_train_count = cv.fit_transform(x_train)

#creating and training the model
model = MultinomialNB()
model.fit(x_train_count, y_train)

#pretesting the model
spam = ['Congratulations!, winner of the cash']
spam_count = cv.transform(spam)
model.predict(spam_count)

# pretesing
ham = ['I love you']
ham_count = cv.transform(ham)
model.predict(ham_count)

#Testing the model

#transform x_test
x_test_count = cv.transform(x_test)

#testing the model proper
model.score(x_test_count, y_test) #returns the mean accuracy on the given test data and labels

# predicting message
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result
 
# Building web application
st.header("Spam Detection")
input_mess = st.text_input('Enter message here')

if st.button("Validate"):
            output = predict(input_mess)
            st.write(output)





