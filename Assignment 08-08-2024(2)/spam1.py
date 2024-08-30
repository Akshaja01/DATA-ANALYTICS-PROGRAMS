

import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics as mat


st.set_page_config(page_title="SPAM DETECTION - EDA",page_icon=":clown_face:",layout="wide")

st.title(":clown_face: SPAM DATA - EDA")
sdf=pd.read_csv("spam.csv", encoding="latin1")
#st.header("spam dataset")
#st.dataframe(sdf.head())
le=LabelEncoder()


sdf.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
st.dataframe(sdf.head())

st.subheader("Statistical summary of dataset")
st.table(sdf.describe())

st.subheader("Changing labels to numerical value")
sdf['label'] = sdf.v1.map({'ham':0, 'spam':1})
st.dataframe(sdf.head())

xtrain, xtest, ytrain, ytest = tts(sdf['v2'], sdf['label'], test_size=0.2, random_state=0)

c1,c2 = st.columns(2)
c1.subheader("Training data size")
c1.write(xtrain.shape)
c1.write(ytrain.shape)

c2.subheader("Testing data size")
c2.write(xtest.shape)
c2.write(ytest.shape)



cv = CountVectorizer()

training_features = cv.fit_transform(xtrain)
test_features = cv.transform(xtest)


mnb = MultinomialNB()
bnb = BernoulliNB()

mnb.fit(training_features, ytrain)
bnb.fit(training_features, ytrain)

ypred_b1 = mnb.predict(test_features)
ypred_b2 = bnb.predict(test_features)


col1,col2 = st.columns(2)
col1.subheader("Accuracy of MultinomialNB")
col1.write(round(mat.accuracy_score(ypred_b1, ytest) *100, 2))

col2.subheader("Accuracy of BernoulliNB")
col2.write(round(mat.accuracy_score(ypred_b2, ytest) *100, 2))

col3,col4 = st.columns(2)
col3.subheader("Confusion Matrix of MultinomialNB")
cm1 = mat.confusion_matrix(ytest, ypred_b1, labels=[0,1])
fig1 = px.imshow(cm1, text_auto=True, labels=dict(x="Predicted values", y="Actual Values"), x=['Spam', 'Ham'], y=['Spam', 'Ham'])
col3.plotly_chart(fig1)

col4.subheader("Confusion Matrix of BernoulliNB")
cm2 = mat.confusion_matrix(ytest, ypred_b2, labels=[0,1])
fig2 = px.imshow(cm2, text_auto=True, labels=dict(x="Predicted values", y="Actual Values"), x=['Spam', 'Ham'], y=['Spam', 'Ham'])
col4.plotly_chart(fig2)


def model1():
    return mnb

def model2():
    return bnb
    
def count_vect():
    return cv
    














