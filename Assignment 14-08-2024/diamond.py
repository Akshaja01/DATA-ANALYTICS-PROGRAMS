import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


st.set_page_config(page_title="Diamond data analysis",page_icon=" :tulip: ",layout="wide")
st.title(" :tulip: Diamond data analysis :tulip:

df=pd.read_csv('diamonds.csv')
st.header('diamond dataset')
st.table(df.head())

st.header("stasticval summary")
st.table(df.describe())

st.header("data visulauization")
fig1=px.scatter(df,x="carat",y="price",color="cut",color_continuous_scale='Virdis')
st.plotly_chart(fig1,use_container_width=True)


st.header("price distribution")
fig2=px.scatter(df,x="price",rbins=20)
st.plotly_chart(fig2,use_container_width=True)
