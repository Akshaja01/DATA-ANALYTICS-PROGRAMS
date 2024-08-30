import streamlit as st
import pandas as pd
import plotly.express as px
st.title("Maps")
file1=st.file_uploader("choose a csv file")
st.dataframe(file1)
if file1:
    df=pd.read_csv(file1)
    st.write(df)

