import streamlit as st
import pickle


model1=pickle.load(open('kmean.pkl','rb'))

st.header("Prediction")
c1, c2 = st.columns(2)
n1=int(c1.number_input("Enter mass value"))
n2=int(c1.number_input("Enter width"))
n3=int(c2.number_input("Enter height"))
n4=int(c2.number_input("Enter color score"))

st.write("Sample values: mass, width, height, color_score")
st.write("Sample values: apple : 180, 8, 6.8, 0.59")
st.write("Sample values: mandarin: 76, 5.8, 4, 0.81")
st.write("Sample values: orange : 154, 7.2, 7.2, 0.82")
st.write("Sample values: lemon : 118, 6.1, 8.1, 0.7")

sample=[[n1,n2,n3,n4]]

if(st.button("Predict the Fruit Label")):
    t = model1.predict(sample)
    if (t == 0):
        st.write(":apple: Apple")
    elif (t == 1):
        st.write(":tangerine: Mandarin")
    elif (t == 2):
        st.write(":tangerine: Orange")
    elif (t == 3):
        st.write(":lemon: Lemon")        
    else:
        st.write("Fruit not listed")
         