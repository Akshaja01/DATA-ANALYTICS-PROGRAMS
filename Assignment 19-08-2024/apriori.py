import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import plotly.express as px
from mlxtend.frequent_patterns import association_rules,apriori

st.set_page_config(page_title="Association Rule Mining", page_icon="🍞😋", layout="wide")
st.title("🍞😋Association Rule Mining  🍞😋")


df=pd.read_csv('bread.csv')

st.header("Sample transaction dataset")
st.table(df.head())
st.divider()

st.header("Finding count of each items in transaction")
tran_df=df.groupby(['Transaction','Item'])