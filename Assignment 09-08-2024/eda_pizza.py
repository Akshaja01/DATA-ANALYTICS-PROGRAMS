import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


st.set_page_config(page_title="Pizza Price", page_icon=":pizza:",layout="wide")
st.title(":pizza: Pizza Data Analysis :pizza:")

df=pd.read_csv('pizza.csv')
st.dataframe(df)

c1,c2=st.columns(2)

c1.header("Relation analysis of features")
fig1=px.scatter(df,x=df.diameter,y=df.rupee)
c1.plotly_chart(fig1,use_container_width=True)


c2.header("Correlation matrix of features")
corr=df.corr()
fig2=px.imshow(corr,text_auto=True,labels=dict(x="Diameter",y="Price"))
c2.plotly_chart(fig1,use_container_width=True)


x=df[['diameter']]
y=df[['rupee']]

lmodel=LinearRegression()

lmodel.fit(x,y)
save_model=pickle.dump(lmodel,open('pizza1.pk1','wb'))
