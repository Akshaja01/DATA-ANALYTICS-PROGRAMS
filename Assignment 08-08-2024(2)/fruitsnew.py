import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="fruits!!!",page_icon=":strawberry:",layout="wide")
st.title(":apple: fruit data analysis!!!")

fdf=pd.read_csv("fruits.csv")

st.subheader(":apple: sample data anlaysis :appple:")

st.header("statistical summary of data")

st.table(fdf['fruit_label'].unique())
st.table(fdf['fruit_name'].unique())

st.header('univariate anlasis')
fig1=px.box(fdf,x='mass',y='fruit_name',title='mass for fruits',color='fruit_name')
st.plotly_chart(fig1,use_container_width=True)

fig2=px.box(fdf,x='width',y='fruit_name',title='width for fruits',color='fruit_name')
st.plotly_chart(fig2,use_container_width=True)


fig3=px.box(fdf,x='width',y='fruit_name',title='mwidth for fruits',color='fruit_name')
st.plotly_chart(fig2,use_container_width=True)





fig4=px.box(fdf,x='color_score',y='fruit_name',title='color for fruits',color='fruit_name')
st.plotly_chart(fig4,use_container_width=True)


def data():
    fdf=pd.read_csv('fruits.csv')
    x=fdf.iloc[:,3:7]
    y=fdf[['fruit_label']]
    return x,y