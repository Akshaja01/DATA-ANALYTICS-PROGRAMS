import streamlit as st

pg=st.navigation([st.Page("airlinearima.py",title="Airline data analysis"),

st.Page("airlinearimapred.py",title="Airline data prediction")])




pg.run()