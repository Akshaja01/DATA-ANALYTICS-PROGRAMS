import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pickle
from sklearn.linear_model import Ridge,Lasso,ElasticNet
import matplotlib.pyplot as plt


st.set_page_config(page_title="Housing",page_icon=":house_buildings:",layout="wide")

st.title(":house_buildings: Housing Data Analysis :house_buildings: ")
sdf = pd.read_csv("Housing.csv")
st.subheader(" Housing Dataset")
st.table(sdf.head())


st.header("null values in data")
st.table(sdf.isnull().sum())

st.header("statstical summary")
st.table(sdf.describe())

st.header("columns of data")
st.subheader(sdf.columns)



le=LabelEncoder()
sdf['mainroad']=le.fit_transform(sdf['mainroad'])
sdf['guestroom']=le.fit_transform(sdf['guestroom'])
sdf['basement']=le.fit_transform(sdf['basement'])
sdf['hotwaterheating']=le.fit_transform(sdf['hotwaterheating'])
sdf['airconditioning']=le.fit_transform(sdf['airconditioning'])
sdf['prefarea']=le.fit_transform(sdf['prefarea'])
sdf['furnishingstatus']=le.fit_transform(sdf['furnishingstatus'])

training_data=sdf[sdf['price'].isnull()==False]
testing_data=sdf[sdf['price'].isnull()==False]


c1,c2=st.columns(2)

c1.subheader("shape of treainibg data")
c1.write(training_data.shape)
c1.subheader("nullvalue of treainibg data")
c1.write(training_data.isnull().sum())
c1.subheader(" treainibg data")
c1.table(training_data.head())



c2.subheader("shape of testing data")
c2.write(testing_data.shape)
c2.subheader("nullvalue of testing data")
c2.write(testing_data.isnull().sum())
c2.subheader(" testing data")
c2.table(testing_data.head())



xtrain=training_data.drop("price",axis=1)
ytrain=training_data[['price']]

xtest=testing_data.drop("price",axis=1)
ytest=testing_data[['price']]



c3,c4,c5,c6=st.columns(4)
c3.subheader("features of training data")
c3.table(xtrain.head())
c4.subheader("labels of training data")
c4.table(xtrain.head())
c5.subheader("features of testing data")
c5.table(xtest.head())
c6.subheader("labels of testing data")
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

st.header("comparoson of different models")


st.subheader("r2 score")
r21=metrics.r2_score(ypred1,ypred2)
r22=metrics.r2_score(ypred1,ypred3)
r23=metrics.r2_score(ypred2,ypred3)
col1,col2,col3=st.columns(3)
col1.write(r21)
col2.write(r22)
col3.write(r23)

st.subheader("mse")
mse1=metrics.mean_squared_error(ypred1,ypred2)
mse2=metrics.mean_squared_error(ypred1,ypred3)
mse3=metrics.mean_squared_error(ypred2,ypred3)
col1,col2,col3=st.columns(3)
col1.write(mse1)
col2.write(mse2)
col3.write(mse3)

st.subheader("mae")
mae1=metrics.mean_absolute_error(ypred1,ypred2)
mae2=metrics.mean_absolute_error(ypred1,ypred3)
mae3=metrics.mean_absolute_error(ypred2,ypred3)
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
