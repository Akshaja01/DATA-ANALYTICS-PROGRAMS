import streamlit as st

pg=st.navigation([
st.Page("association_eda.py",title="Association rule mining"),
st.Page("wine.py",title="Wine Data Analysis"),
st.Page("airline.py",title="AirLine Data Analysis"),
st.Page("breast.py",title="Breast Cancer Data Analysis"),
])

pg.run()
