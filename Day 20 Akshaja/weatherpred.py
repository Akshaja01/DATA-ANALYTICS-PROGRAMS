
import streamlit as st
import pickle

gnb = pickle.load(open('weather.pkl', 'rb'))

st.header("Prediction")
n1=int(st.number_input("Enter overcast value"))
n2=int(st.number_input("Enter temperature value"))
n3=int(st.number_input("Enter humidity value"))
n4=int(st.number_input("Enter windy  value"))

sample1=[[n1,n2,n3,n4]]

if st.button("Predict good weather to play cricket"):
    target_sp=gnb.predict(sample1)
    st.write(target_sp)
    if (target_sp == 1):
        st.write("Weather good for playing Cricket")
    else:
        st.write("Not good for playing")