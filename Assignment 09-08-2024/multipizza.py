import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

model=pickle.load(open('pizza2.pk1', 'rb'))
n1=int(st.number_input("Do you need extra cheeze"))
n2=int(st.number_input("Do you need extra mushroom"))
n3=int(st.number_input("Size of pizza"))
n4=int(st.number_input("Do you need extra spices"))

new_feature=[[n1,n2,n3,n4]]
if st.button("Predict the price"):
    t2=model.predict(new_feature)
    st.subheader("Predicted price is")
    st.write(t2)