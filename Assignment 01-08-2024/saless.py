import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.header("Company sales data")
df3=pd.read_csv('company_sales_data.csv')
st.write(df3)
st.header("a.line plot")
st.line_chart(df3,x='month_number',y='total_profit')

st.header("b.multi line plot")
products=['facecream','facewash','toothpaste','bathingsoap','shampoo','moisturizer']
st.line_chart(df3,x="month_number",y=products)

st.header("c.scatter plot")
st.scatter_chart(df3,x="month_number",y="toothpaste")

st.header("c.bar plot")


st.header("c.pie plot")
products=['facecream','facewash','toothpaste','bathingsoap','shampoo','moisturizer']
total_sales=df3[products].sum()
products_sum=[]
for i in products:
    products_sum.append(i.title()+"\n"+str(df3[i].sum())+"units")
explode=[0,0.2,0,0,0,0]
fig1,ax1=plt.subplots()
ax1.pie(total_sales,explode=explode,labels=products_sum,autopct='%1.1f%%')
ax1.axis('equal')
st.pyplot(fig1)
