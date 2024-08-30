import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


st.set_page_config(page_title="breast cancer analysis",page_icon=":hospital:",layout="wide")

st.title(" :stethoscope: Breast cancer   :stethoscope: ")

data=load_breast_cancer()
bdata=data.data
blabel=data.target
st.write(data)
st.header("featurs of s=dataset")


st.write(data.feature_names)

c1,c2=st.columns(2)
c1.subheader("size of")
c1.write(bdata.shape)

c2.subheader("size of labels")
st.header("stastical summary")
st.write(data.DESCR)


xtrain,xtest,ytrain,ytest=tts(bdata,blabel,test_size=0.25,random_state=10)

logmodel=LogisticRegression()

logmodel.fit(xtrain,ytrain)

ypred=logmodel.predict(xtest)


st.header("perfromance measure model")

mse=metrics.mean_squared_error(ypred,ytest)

r2=metrics.r2_score(ypred,ytest)
mae=metrics.mean_absolute_error(ypred,ytest)

c3,c4,c5=st.columns(3)

c3.subheader("mean squared error")

c3.write(mse)

c4.subheader("r2score")

c4.write(r2)


c5.subheader("mean absolute error")

c5.write(mae)

st.write(bdata)

