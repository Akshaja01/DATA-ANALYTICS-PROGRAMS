

import streamlit as st
import pickle
st.subheader(":sandwich: Obesity Prediction :pizza:")

model1=pickle.load(open('obesity_kmeans.pkl','rb'))

st.header("Prediction")
c1, c2 = st.columns(2)


n1=int(c1.number_input("Enter Age"))
n2=int(c1.number_input("Enter Gender"))
n3=int(c2.number_input("Enter Height"))
n4=int(c2.number_input("Enter Weight"))
n5=int(c2.number_input("Enter BMI"))


st.write("Sample values: Age, gender, height, weight, bmi")
st.write("Sample values: 25	1	175	80	25.3	Normal Weight")
st.write("Sample values: 30	0	160	60	22.5	Normal Weight")
st.write("Sample values: 35	1	180	90	27.3	Overweight")
st.write("Sample values: 40	0	150	50	20	Underweight")
st.write("Sample values: 45	1	190	100	31.2	Obese")


sample=[[n1,n2,n3,n4,n5]]

if(st.button("Predict the Obesity level")):
    t = model1.predict(sample)
    if (t == 0):
        st.write("Normal Weight")
    elif (t == 1):
        st.write("Underweight")
    elif (t == 2):
        st.write("Overweight")
    elif (t == 3):
        st.write("Obese")        
    else:
        st.write("Cannot determine")
         