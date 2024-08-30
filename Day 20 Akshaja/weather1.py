
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn import metrics as mat
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as mat
import pickle


st.set_page_config(page_title="Cricket Times",page_icon=":cricket_bat_and_ball:",layout="wide")

st.title(":sun_behind_cloud: Weather Data Analysis :cricket_bat_and_ball: ")
wdf = pd.read_csv("weather.csv")
st.subheader("Weather Dataset")
st.dataframe(wdf.head())

def encoding():
    le=LabelEncoder()
    wdf['outlook']=le.fit_transform(wdf['outlook'])
    wdf['temperature']=le.fit_transform(wdf['temperature'])
    wdf['humidity']=le.fit_transform(wdf['humidity'])
    wdf['windy']=le.fit_transform(wdf['windy'])
    wdf['play']=le.fit_transform(wdf['play'])

    st.dataframe(wdf.head())
    return wdf

encoding()
st.header("Checking for Presence of null values")
st.write(wdf.isnull().sum())


x = wdf.drop('play', axis=1)
y = wdf[['play']]

st.write(x)
st.write(y)

c1, c2 = st.columns(2)
c1.header("Features shape") 
c1.write(x.shape)
c2.header("Label shape") 
c2.write(y.shape)


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state = 10)

cl1,cl2,cl3,cl4 = st.columns(4)
cl1.subheader("Training Features shape")
cl2.subheader("Training Labels shape")
cl3.subheader("Testing Features shape")
cl4.subheader("Testing Labels shape")
cl1.write(xtrain.shape)
cl2.write(ytrain.shape)
cl3.write(xtest.shape)
cl4.write(ytest.shape)


gnb   = GaussianNB()
gnb.fit(xtrain, ytrain)
ypred = gnb.predict(xtest)


st.header("Accuracy of the model")
st.write(mat.accuracy_score(ytest,ypred) * 100)

cm   = mat.confusion_matrix(ytest, ypred, labels =[0,1])
disp = px.imshow(cm, text_auto=True, labels=dict(x="Predicted values", y="Actual values"), x=['Yes','No'], y=['Yes','No'])
st.plotly_chart(disp, use_container_width=True)

p = pickle.dump(gnb, open('weather.pkl','wb'))
















