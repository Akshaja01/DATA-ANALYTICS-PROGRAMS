import streamlit as st
from datetime import *
st.title("Registration form")
name=st.text_input("Name")
gender=st.radio("Gender",("Male","Female"))
phoneno=st.text_input("Phonenumber")
dob=st.date_input("Date of Birth")
if st.button("Submit"):
    st.write("Name",name)
    st.write("Gender",gender)
    st.write("Phonenumber",phoneno)
    st.write("Date of Birth",dob)
