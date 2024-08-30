import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as mat
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Obesity!!!",page_icon=":sandwich:",layout="wide")
st.title(":pizza: Obesity Data Analysis!!!")

df=pd.read_csv("obesity.csv")

st.table(df.head())
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df


x = df.iloc[:,1:6]  


agg1=AgglomerativeClustering(n_clusters=4,linkage='single')
agg2=AgglomerativeClustering(n_clusters=4,linkage='complete')
agg3=AgglomerativeClustering(n_clusters=4,linkage='average')

df['new_label1']=agg1.fit_predict(x)
df['new_label2']=agg2.fit_predict(x)
df['new_label3']=agg3.fit_predict(x)

st.header("Visualizing the new label1 and clusters")
fig1=px.scatter(df,x='Age',y='Weight',size='BMI',color='new_label1')
st.plotly_chart(fig1,use_container_width=True)

st.header("Visualizing the new label2 and clusters")
fig2=px.scatter(df,x='Age',y='Weight',size='BMI',color='new_label2')
st.plotly_chart(fig2,use_container_width=True)

st.header("Visualizing the new label3 and clusters")
fig3=px.scatter(df,x='Age',y='Weight',size='BMI',color='new_label3')
st.plotly_chart(fig3,use_container_width=True)

dbs1=mat.davies_bouldin_score(x,df['new_label1'])
sil1=mat.silhouette_score(x,df['new_label1'])
cal1=mat.calinski_harabasz_score(x,df['new_label1'])

st.header("Evaluation score of single linkage agglomerative")

c1,c2,c3=st.columns(3)

c1.subheader('Davies_bouldin_score')
c1.subheader(dbs1)
c2.subheader('silbouette_score')
c2.subheader(sil1)
c3.subheader('calinski_harabasz_score')
c3.subheader(cal1)

dbs2=mat.davies_bouldin_score(x,df['new_label2'])
sil2=mat.silhouette_score(x,df['new_label2'])
cal2=mat.calinski_harabasz_score(x,df['new_label2'])

st.header("Evaluation score of complete linkage agglomerative")

c1,c2,c3=st.columns(3)

c1.subheader('Davies_bouldin_score')
c1.subheader(dbs2)
c2.subheader('silbouette_score')
c2.subheader(sil2)
c3.subheader('calinski_harabasz_score')
c3.subheader(cal2)

dbs3=mat.davies_bouldin_score(x,df['new_label3'])
sil3=mat.silhouette_score(x,df['new_label3'])
cal3=mat.calinski_harabasz_score(x,df['new_label3'])

st.header("Evaluation score of average linkage agglomerative")

c1,c2,c3=st.columns(3)

c1.subheader('Davies_bouldin_score')
c1.subheader(dbs3)
c2.subheader('silbouette_score')
c2.subheader(sil3)
c3.subheader('calinski_harabasz_score')
c3.subheader(cal3)