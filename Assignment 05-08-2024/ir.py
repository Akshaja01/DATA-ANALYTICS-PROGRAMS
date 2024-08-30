import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris=sns.load_dataset('iris')
st.title("Iris Data Analysis")
st.subheader("Iris Dataset")
st.dataframe(iris)
st.subheader("Summary statistics")
st.write(iris.describe())
st.subheader("Pairplot")
pairplot=sns.pairplot(iris,hue='species')
st.pyplot(pairplot)
plt.figure(figsize=(10,6))
heatmap=sns.heatmap(iris.corr(),annot=True,cmap='coolwarm')
st.pyplot(heatmap.figure)