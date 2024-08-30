import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics as mat
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster,linkage

st.set_page_config(page_title="Divisive clustering in obesity data", page_icon="🍇", layout="wide")
st.title("🍉 Divisive clustering in obesity data🍉")

df=pd.read_csv('fruits.csv')
x=df.iloc[:,3:7]

distance=linkage(x,method='single')

#df['clust1']=fcluster(distance,1,criterion='maxclust')
df['clust2']=fcluster(distance,2,criterion='maxclust')
df['clust3']=fcluster(distance,3,criterion='maxclust')
df['clust4']=fcluster(distance,4,criterion='maxclust')

# fig1=px.scatter(df,x='width',y='height',color='clust1')
# st.plotly_chart(fig1)

fig2=px.scatter(df,x='width',y='height',color='clust2')
st.plotly_chart(fig2)

fig3=px.scatter(df,x='width',y='height',color='clust3')
st.plotly_chart(fig3)

fig4=px.scatter(df,x='width',y='height',color='clust4')
st.plotly_chart(fig4)

# dbs1=mat.davies_bouldin_score(x,df['clust1'])
# sil1=mat.silhouette_score(x,df['clust1'])
# cal1=mat.calinski_harabasz_score(x,df['clust1'])


# st.header("Evaluation score for cluster 1 ")

# c1,c2,c3=st.columns(3)

# c1.subheader('Davies_bouldin_score')
# c1.subheader(dbs1)
# c2.subheader('silbouette_score')
# c2.subheader(sil1)
# c3.subheader('calinski_harabasz_score')
# c3.subheader(cal1)

dbs2=mat.davies_bouldin_score(x,df['clust2'])
sil2=mat.silhouette_score(x,df['clust2'])
cal2=mat.calinski_harabasz_score(x,df['clust2'])


st.header("Evaluation score for cluster 2 ")

c1,c2,c3=st.columns(3)

c1.subheader('Davies_bouldin_score')
c1.subheader(dbs2)
c2.subheader('silbouette_score')
c2.subheader(sil2)
c3.subheader('calinski_harabasz_score')
c3.subheader(cal2)


dbs3=mat.davies_bouldin_score(x,df['clust3'])
sil3=mat.silhouette_score(x,df['clust3'])
cal3=mat.calinski_harabasz_score(x,df['clust3'])


st.header("Evaluation score for cluster 3 ")

c1,c2,c3=st.columns(3)

c1.subheader('Davies_bouldin_score')
c1.subheader(dbs3)
c2.subheader('silbouette_score')
c2.subheader(sil3)
c3.subheader('calinski_harabasz_score')
c3.subheader(cal3)