
import streamlit as st
import pandas as pd
import plotly.express as px
import sklearn.metrics as mat
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Cancer Data Analysis", page_icon="👩🏻‍⚕️🎗️", layout="wide")
st.title("👩🏻‍⚕️🎗️Breast Cancer Data Analysis️ 👩🏻‍⚕️🎗️")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

st.header('BREAST CANCER DATA')
st.table(df.head())


st.header("Statistical Summary of Data set")
st.table(df.describe())

st.header("Visualizing labels and clusters")
fig1 = px.scatter(df, x="mean radius", y="mean texture", size="mean perimeter", color="diagnosis")
st.plotly_chart(fig1, use_container_width=True)


x = df.drop(['diagnosis'], axis=1)
kmeans = KMeans(n_clusters = 3, init = "k-means++", max_iter = 30, random_state=0)
df['y1'] = kmeans.fit_predict(x)

st.header("Applying PCA on features")
x_scaled = scale(x)
pca      = PCA(n_components = 6)
pca_x    = pca.fit_transform(x_scaled)


df['y2'] = kmeans.fit_predict(pca_x)

st.header("Visualizing the labels without PCA and clusters") 
fig2 = px.scatter(df, x="mean radius", y="mean texture", size="mean perimeter", color="y1")
st.plotly_chart(fig2, use_container_width=True)


st.header("Visualizing the labels with PCA and clusters")
fig3 = px.scatter(df, x="mean radius", y="mean texture", size="mean perimeter", color="y2")
st.plotly_chart(fig3, use_container_width=True)

c1, c2 = st.columns(2)
c1.subheader("Silhouette Score without PCA")
c1.write(mat.silhouette_score(x, df['y1']))
c2.subheader("Silhouette Score with PCA")
c2.write(mat.silhouette_score(x, df['y2']))

