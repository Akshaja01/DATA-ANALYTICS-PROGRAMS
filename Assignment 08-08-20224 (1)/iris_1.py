import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn
import streamlit as st

st.set_page_config(page_title="IRIS WEBAPP", page_icon=":tulip:",layout="wide")

st.title(":tulip: Iris Data Analysis")

iris=pd.read_csv('Iris.csv')
st.header("Iris Dataset")
st.dataframe(iris.head())

st.subheader("Univariate Analysis")
col1,col2=st.columns(2)

fig1=px.box(iris,x='SepalLengthCm',y='Species',title='Sepal Length of species')
col1.plotly_chart(fig1,use_container_width=True)
fig2=px.box(iris,x='SepalWidthCm',y='Species',title='Sepal Width of species')
col2.plotly_chart(fig2,use_container_width=True)
fig3=px.box(iris,x='PetalLengthCm',y='Species',title='Petal Length of species')
col1.plotly_chart(fig3,use_container_width=True)
fig4=px.box(iris,x='PetalWidthCm',y='Species',title='Petal Width of species')
col2.plotly_chart(fig4,use_container_width=True)

st.subheader("Bivariant Analysis")
fig5=sns.pairplot(iris,hue='Species')
st.pyplot(fig5.fig