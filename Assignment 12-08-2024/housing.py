import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge,Lasso,ElasticNet
import matplotlib.pyplot as plt
import pickle



st.set_page_config(page_title="Hitters",page_icon=":house_buildings:",layout="wide")

st.title(":house_buildings: Hitter Data Analysis :house_buildings: ")
sdf = pd.read_csv("Housing.csv")
st.subheader("Housing Dataset")
st.table(sdf.head())

st.subheader("Null values in data")
st.table(sdf.isnull().sum())
st.header("Statistical summary of data")
st.table(sdf.describe())

st.header("Metadata information of data")
st.table(sdf.info())

st.header("Column of data")
st.subheader(sdf.columns)

le=LabelEncoder()
sdf['mainroad']=le.fit_transform(sdf['mainroad'])
sdf['guestroom']=le.fit_transform(sdf['guestroom'])
sdf['basement']=le.fit_transform(sdf['basement'])
sdf['hotwaterheating']=le.fit_transform(sdf['hotwaterheating'])
sdf['airconditioning']=le.fit_transform(sdf['airconditioning'])
sdf['furnishingstatus']=le.fit_transform(sdf['furnishingstatus'])



training_data=sdf[sdf['price'].isnull()==False]
testing_data=sdf[sdf['price'].isnull()==False]

c1,c2=st.columns(2)
c1.subheader("Shape of training data")
c1.write(training_data.shape)
c1.subheader("Null values in training data")
c1.write(training_data.isnull().sum())
c1.subheader("training data")
c1.table(training_data.head())

c2.subheader("Shape of testing data")
c2.write(testing_data.shape)
c2.subheader("Null values in testingg data")
c2.write(testing_data.isnull().sum())
c2.subheader("testing data")
c2.table(testing_data.head())

xtrain=training_data.drop('price',axis=1)
ytrain=training_data[['price']]

xtest=testing_data.drop('price',axis=1)
ytest=testing_data[['price']]

c3,c4,c5,c6=st.columns(4)
c3.subheader("Features of training data")
c3.table(xtrain.head())

c4.subheader("Labels of training data")
c4.table(ytrain.head())

c5.subheader("Features of testing data")
c5.table(xtest.head())

c6.subheader("Labels of testing data")
c6.table(ytest.head())


rid=Ridge()
lass=Lasso()
enet=ElasticNet()

rid.fit(xtrain,ytrain)
lass.fit(xtrain,ytrain)
enet.fit(xtrain,ytrain)

m1=pickle.dump(rid,open('rid1.pkl','wb'))
m2=pickle.dump(lass,open('lass1.pkl','wb'))
m3=pickle.dump(enet,open('enet1.pkl','wb'))

ypred1=rid.predict(xtest)
ypred2=lass.predict(xtest)
ypred3=enet.predict(xtest)

st.header("Comparison of different models")
st.subheader('R2 score')

r21=metrics.mean_squared_error(ypred1,ypred2)
r22=metrics.mean_squared_error(ypred1,ypred3)
r23=metrics.mean_squared_error(ypred2,ypred3)

col1,col2,col3=st.columns(3)
col1.write(r21)
col2.write(r22)
col3.write(r23)

st.subheader("MSE")
mse1=metrics.mean_squared_error(ypred1,ypred2)
mse2=metrics.mean_squared_error(ypred1,ypred2)
mse3=metrics.mean_squared_error(ypred1,ypred2)

col1,col2,col3=st.columns(3)
col1.write(mse1)
col2.write(mse2)
col3.write(mse3)

st.subheader("MAE")
mae1=metrics.mean_absolute_error(ypred1,ypred2)
mae2=metrics.mean_absolute_error(ypred1,ypred2)
mae3=metrics.mean_absolute_error(ypred1,ypred2)

col1,col2,col3=st.columns(3)
col1.write(mae1)
col2.write(mae2)
col3.write(mae3)

st.header("Prediction of different models")
testing_data['Ridge_Salary']=ypred1
testing_data['Lasso_Salary']=ypred2
testing_data['Enet_Salary']=ypred3

fig,ax=plt.subplots(figsize=(2,2))
ax.plot(ypred1,c='g',marker='+')
ax.plot(ypred2,c='b',marker='+')
ax.plot(ypred3,c='r',marker='+')

st.pyplot(fig)