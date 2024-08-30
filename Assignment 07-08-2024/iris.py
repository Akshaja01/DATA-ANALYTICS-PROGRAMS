import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
import  seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn.import metrics mat

st.set_page_config(page_title="Machine Learning with Iris",page_icon=":tulip:",layout="wide")

st.title(":tulip: Iris Data Analysis")
Iris=pd.read_csv('Iris.csv')

st.subheader("Iris Dataset")
st.dataframe(Iris.head())

x=Iris.drop(columns['Species'],axis=0)
y=Iris.['Species']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=10)
model=dtc(criterion='entropy')
model.fit(xtrain,ytrain)
samples=[(6.2,3.4,5.4,2.3)]
target_species=model.predict(sample1)
st.set_page=config
ypred=model.predict(xtest)
accuracy_Iris=mat.accuracy_score(ytest,ypred)
precision_Iris=mat.precision_score(ytest,ypred)
col1,col2=st.columns(2)
col1.subheader("Accuracy score for iris model")
col1.subheader(accuracy_Iris)
col2.subheader("Precision score for iris model")
col2.subheader(precision_Iris)
cm=mat.confusion_matrix(ytest,ypred)
fig1=px.imshow(cm,test_auto=True)
st.plotly_chart(fig1.use_container_width=True)
st.subheader("Classification Report for iris model")
st.subheader(mat.classification_report(ytest,ypred)