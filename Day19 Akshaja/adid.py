import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from datetime import *
from geopy.geocoders import Nominatim


warnings.filterwarnings('ignore')


st.set_page_config(page_title="Adidas United States Dashboard!!!",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: Adidas United States Dashboard!!!!!!")

df=pd.read_excel("adidas.xlsx",sheet_name='Sales')
st.subheader("first 5 rows of adidas data")
st.dataframe(df.head())

df['Years']=pd.to_datetime(df['InvoiceDate']).dt.year
df['Months']=pd.to_datetime(df['InvoiceDate']).dt.month

st.sidebar.header("Choose your filter: ")
Year=st.sidebar.multiselect("select the year",df['Years'].unique())
Region=st.sidebar.multiselect("Pick your region",df["Region"].unique())
SalesMethod=st.sidebar.multiselect("Pick method of sales",df["SalesMethod"].unique())

col1,col2,col3=st.columns(3)

col1.subheader("Total sales")
col1.write(df['TotalSales'].sum())

col2.subheader("Total units sold")
col2.write(df['UnitsSold'].sum())

col3.subheader("Total Profit")
col3.write(df['OperatingProfit'].sum())

#table1=pd.pivot_table(df,values='Sales',index=['Category'],aggfunc=np.sum).reset_index()


st.subheader("Total sales Trend")
table1=pd.pivot_table(df,values='TotalSales',index=df['Months'],aggfunc=np.sum).reset_index()
fig1=px.line(table1,x="Months",y="TotalSales")
st.plotly_chart(fig1,use_container_width=True)


st.subheader("Total profit Trend")
table2=pd.pivot_table(df,values='OperatingProfit',index=df['Months'],aggfunc=np.sum).reset_index()
fig2=px.line(table2,x="Months",y="OperatingProfit")
st.plotly_chart(fig2,use_container_width=True)

col3,col4=st.columns(2)

st.subheader("sales by product")
table3=pd.pivot_table(df,values='TotalSales',index=['Product'],aggfunc=np.sum).reset_index()
t3=table3.sort_values(by="TotalSales",ascending=True)
fig3=px.funnel(t3,x='TotalSales',y='Product',color="Product")
st.plotly_chart(fig3,use_container_width=True)

st.subheader("sales by retailer")
table4=pd.pivot_table(df,values='TotalSales',index=['Retailer'],aggfunc=np.sum).reset_index()
fig4=px.pie(table4,values="TotalSales",names="Retailer",hole=0.2)
st.plotly_chart(fig4,use_container_width=True)
    
    
st.subheader("Top 10 sales by state") 
 
table5=pd.pivot_table(df,values='TotalSales',index=df['State'],aggfunc=np.sum).reset_index()
fig5=px.line(table5,x="State",y="TotalSales")
st.plotly_chart(fig5,use_container_width=True,orientation='h')


geolocator=Nominatim(user_agent="geoapiExercises")
tab1=pd.pivot_table(df,values='TotalSales',index=['State'],aggfunc=np.sum).reset_index()

st.table(tab1)
st.dataframe(tab1.head())

def get_lat_lon(State):
    location=geolocator.geocode(State)
    return location.latitude,location.longitude
tab1['latitude'],tab1['longitude']=zip(*tab1['State'].apply(get_lat_lon))
st.dataframe(tab1.head())


st.map(tab1,latitude='latitude',longitude='longitude')