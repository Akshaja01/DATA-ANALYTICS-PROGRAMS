import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn .svm import LinearSVC


st.set_page_config(page_title="Iris data analysis",page_icon=" :tulip: ",layout="wide")
st.title(" :rose: Iris data Analysis :rose: ")


iris=pd.read_csv('iris.csv')
st.header('IRIS DATA')
st.table(iris.head())

le=LabelEncoder()
iris['Label']=le.fit_transform(iris['Species'])
iris.drop(columns=['Id'],axis=1,inplace=True)
st.header('IRIS DATA')
st.table(iris.head())


# Dividing data into x and y

x=iris.drop(columns=['Species','Label'])
y=iris[['Label']]

c1,c2=st.columns(2)

c1.subheader("Features set")
c1.table(x.head())

c2.subheader("Labels")
c2.table(y.head())

xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state = 10,shuffle=True,stratify=y)

c3,c4,c5,c6=st.columns(4)

c3.subheader("Training features size")
c3.table(xtrain.head())

c4.subheader("Training labels size")
c4.table(ytrain.head())

c5.subheader("Testing features size")
c5.table(xtest.head())

c6.subheader("Testing labels size")
c6.table(ytest.head())

linearsvm=LinearSVC()
linearsvm.fit(xtrain,ytrain)

m1=pickle.dump(linearsvm,open('svcm1.pkl','wb'))

ypred=linearsvm.predict(xtest)

st.header("Confusion matrix of the model")

cm=mat.confusion_matrix(ytest,ypred)

disp=px.imshow(cm,text_auto=True,labels=dict(x='predicted values',y='Actual values'),x=[0,1,2],y=[0,1,2])

st.plotly_chart(disp,container_width=True)

st.header("Classification report")

c=mat.classification_report(ytest,ypred,output_dict=True)

st.write(c)
