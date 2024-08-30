import streamlit as st
import pickle

st.set_page_config(page_title="diabetes  prediction",page_icon=" 🍩 ",layout="wide")
st.title("  🍬  diabetes prediction  🍬  ")
model1=pickle.load(open('bag.pkl','rb'))
model2=pickle.load(open('rnd.pkl','rb'))

#  
c1,c2=st.columns(2)
n1=c1.number_input("Pregnancies")
n2=c2.number_input("Glucose")
n3=c1.number_input("BloodPressure")
n4=c2.number_input("SkinThickness")
n5=c1.number_input("Insulin")
n6=c2.number_input("BMI")
n7=c1.number_input("DiabetesPedigreeFunction")
n8=c2.number_input("Age")


new_features=[[n1,n2,n3,n4,n5,n6,n7,n8]]

if st.button("Predict"):
    c5,c6=st.columns(2)
    t1=model1.predict(new_features)
    c5.subheader("Result 1 ")
    if t1==1:
        c5.write("Has diabetes")
    elif t1==0:
        c5.write("Does not have diabetes")
    else:
        c5.write("Cannot predict")
   
    t1=model2.predict(new_features)
    c6.subheader("Result 2 ")
    if t1==1:
        c6.write("Has diabetes")
    elif t1==0:
        c6.write("Does not have diabetes")
    else:
        c6.write("Cannot predict")