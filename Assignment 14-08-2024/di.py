import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title="Diamond Data Analysis", page_icon="💎", layout="wide")
st.title("💎 Diamond Data Analysis 💎")

df=pd.read_csv('diamonds.csv')
st.header('diamond dataset')
st.table(df.head())

st.header("stastistical summary")
st.table(df.describe())

st.header("data visulauization")
fig1=px.scatter(df,x="carat",y="price",color="cut",color_continuous_scale='Virdis')
st.plotly_chart(fig1,use_container_width=True)


st.subheader("price distribution")
fig2=px.histogram(df,x="price",nbins=20)
st.plotly_chart(fig2,use_container_width=True)




st.header("box plot for price with cuts")
fig3=px.box(df,x="cut",y="price",color="cut")
st.plotly_chart(fig3,use_container_width=True)


#cat_col=['cut','clarity','color']
le=LabelEncoder()
#for col in cat_col:
    #df[col]=le.fit_transform(df[col])
st.header("updated dimaond dataset")
st.table(df.head())

x=df.drop(columns=['price'],axis=1)
y=df[['price']]


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