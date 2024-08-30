import streamlit as st
import time
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Analytics Dashboard!!!",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: HR Analytics Dashboard!!!")

df=pd.read_excel('hrdata.xlsx',sheet_name='HR-Data')
st.subheader("First 5 rows of HR data")
st.dataframe(df.head())

#st.subheader("Attrition rate based of age-band for HR data")

col1,col2=st.columns(2)

st.sidebar.header("Choose your filter: ")
gender=st.sidebar.multiselect("Pick your gender",df["Gender"].unique())


if not gender:
    st.warning("Select a gender?......")
else:
    col1.subheader("Attrition rate based of age-band for HR data")
    df1=df[df["Gender"].isin(gender)]
    #st.dataframe(df1.head())
    table1=pd.pivot_table(df1,values='CF_attrition count',index=['CF_age band'],aggfunc=np.sum).reset_index()
    #st.table(table1)
    fig1=px.bar(table1,x="CF_attrition count",y="CF_age band",color="CF_age band")
    col1.plotly_chart(fig1,use_container_width=True)
    
    col2.subheader("Attrition rate based of Marital Status for HR data")
    table2=pd.pivot_table(df1,values='CF_attrition count',index=['Marital Status'],aggfunc=np.sum).reset_index()
    fig2=px.pie(table2,values="CF_attrition count",names="Marital Status",hole=0.5)
    col2.plotly_chart(fig2,use_container_width=True)
    
    col1.subheader("Attrition rate based of Education for HR data")
    table3=pd.pivot_table(df1,values='CF_attrition count',index=['Education'],aggfunc=np.sum).reset_index()
    fig3=px.bar(table3,x="CF_attrition count",y="Education",orientation='h')
    col1.plotly_chart(fig3,use_container_width=True)
    
    col2.subheader("Attrition rate based of Job Roles for HR data")
    table4=pd.pivot_table(df1,values='CF_attrition count',index=['Job Role'],aggfunc=np.sum).reset_index()
    fig4=px.funnel(table4,x='CF_attrition count',y='Job Role')
    col2.plotly_chart(fig4,use_container_width=True)
    
    col1.subheader("Attrition rate based on department for H R data")
    table5=pd.pivot_table(df1,values='CF_attrition count',index=['Department'],aggfunc=np.sum).reset_index()
    fig5=px.pie(table5,values="CF_attrition count",names="Department",hole=0.5)
    col2.plotly_chart(fig5,use_container_width=True)
    
    col2.subheader("Attrition rate based on gender for H R data")
    table6=pd.pivot_table(df1,values='CF_attrition count',index=['Gender'],aggfunc=np.sum).reset_index()
    fig6=px.pie(table6,values="CF_attrition count",names="Gender",hole=0.5)
    col2.plotly_chart(fig6,use_container_width=True)