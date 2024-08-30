import streamlit as st
import pickle

lmodel1=pickle.load(open('pizza1.pk1','rb'))

#lmodel1=pickle.load(saved_model)

st.title(":pizza: Pizza price prediction :pizza:")

d=st.number_input("Enter new diameter of pizza")
d=[[d]]

if st.button("Predict the price"):
    t=lmodel1.predict(d)
    st.subheader("The predicted price of Pizza is")
    st.write(t)