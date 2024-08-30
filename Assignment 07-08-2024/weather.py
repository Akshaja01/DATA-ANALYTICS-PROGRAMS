import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder




from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title="WEATHER DATA PREDICTION",page_icon=":moon:",layout="wide")

st.title(":moon: weather data analysis")
wdf=pd.read_csv("weather.csv")
st.header("weather dataset")
st.dataframe(wdf.head())

le=LabelEncoder()
wdf['outlook']=le.fit_transform(wdf['outlook'])
wdf['temperature']=le.fit_transform(wdf['temperature'])
wdf['humidity']=le.fit_transform(wdf['humidity'])
wdf['windy']=le.fit_transform(wdf['windy'])
wdf['play']=le.fit_transform(wdf['play'])

st.dataframe(wdf.head())

x=wdf.drop('play',axis=1)
y=wdf['play']
wmodel=dtc(criterion='entropy',random_state=0)
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2)
wmodel.fit(xtrain,ytrain)
ypred=wmodel.predict(xtest)
st.header("classification report")
#st.write(mat.classification_report(ytest,ypred))


st.header("prediction")
n1=st.number_input("enter overcast value")
n2=st.number_input("enter temperature value")
n3=st.number_input("enter humidity value")
n4=st.number_input("enter windy  value")

n1=int(n1)
n2=int(n2)
n3=int(n3)
n4=int(n4)

sample1=[[n1,n2,n3,n4]]

if st.button("predict good item to play cricket"):
    target_sp=wmodel.predict(sample1)
    proba=wmodel.predict_proba(sample1)
    st.write(proba)
    st.write(target_sp)